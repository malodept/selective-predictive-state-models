from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class DeltaTransformer(nn.Module):
    def __init__(self, token_dim, action_dim, tokens, model_dim, heads, layers, dropout, mode):
        super().__init__()
        self.mode = mode
        self.tokens = tokens
        self.in_proj = nn.Linear(token_dim, model_dim)
        self.action_proj = nn.Linear(action_dim, model_dim)
        self.pos = nn.Parameter(torch.zeros(1, tokens, model_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, token_dim)

    def forward(self, z_tokens, action):
        if self.mode in ["full", "state_only"]:
            h = self.in_proj(z_tokens)
        else:
            h = torch.zeros(
                z_tokens.shape[0],
                z_tokens.shape[1],
                self.pos.shape[-1],
                device=z_tokens.device,
                dtype=z_tokens.dtype,
            )

        if self.mode in ["full", "action_only"]:
            h = h + self.action_proj(action).unsqueeze(1)

        h = h + self.pos
        h = self.encoder(h)
        h = self.norm(h)
        return self.out_proj(h)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-groups", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.out.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    g = np.load(args.groups, allow_pickle=True)

    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    a = d["action"].astype(np.float32)
    delta = y - z

    anchor = g["anchor_indices"].astype(np.int64)
    cand = g["candidate_indices"].astype(np.int64)
    neg_type = g["negative_type"].astype(str)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]
    mode = ckpt_args["mode"]

    model = DeltaTransformer(
        token_dim=int(ckpt["token_dim"]),
        action_dim=int(ckpt["action_dim"]),
        tokens=int(ckpt["tokens"]),
        model_dim=int(ckpt_args["model_dim"]),
        heads=int(ckpt_args["heads"]),
        layers=int(ckpt_args["layers"]),
        dropout=float(ckpt_args["dropout"]),
        mode=mode,
    ).to(args.device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    n_groups = cand.shape[0]
    rng = np.random.default_rng(int(ckpt_args.get("seed", 0)))
    perm = rng.permutation(n_groups)

    n_train = int(float(ckpt_args.get("train_frac", 0.8)) * n_groups)
    n_val = int(float(ckpt_args.get("val_frac", 0.1)) * n_groups)
    test_groups = perm[n_train + n_val:]

    preds = []
    tgts = []
    typs = []

    for s in range(0, len(test_groups), args.batch_groups):
        gids = test_groups[s:s + args.batch_groups]
        anch = anchor[gids]
        c = cand[gids]

        zc = torch.from_numpy(z[anch]).float().to(args.device)
        aa = torch.from_numpy(a[anch]).float().to(args.device)

        pred = model(zc, aa).cpu().numpy()

        preds.append(pred)
        tgts.append(delta[c])
        typs.append(neg_type[gids])

    pred = np.concatenate(preds, axis=0)
    cand_delta = np.concatenate(tgts, axis=0)
    typ = np.concatenate(typs, axis=0)

    dist = ((pred[:, None] - cand_delta) ** 2).mean(axis=(2, 3))
    pos = dist[:, 0]

    rows = []

    for t in ["same_state_diff_action", "same_action_diff_state"]:
        mask = typ == t
        wins = []
        margins = []

        for i in range(dist.shape[0]):
            js = np.where(mask[i])[0]
            js = js[js != 0]
            for j in js:
                wins.append(pos[i] < dist[i, j])
                margins.append(dist[i, j] - pos[i])

        wins = np.asarray(wins)
        margins = np.asarray(margins)

        rows.append({
            "negative_type": t,
            "pairwise_accuracy": float(wins.mean()),
            "mean_margin": float(margins.mean()),
            "count": int(len(wins)),
        })

    order = np.argsort(dist, axis=1)
    top1 = order[:, 0] == 0
    ranks = np.array([np.where(order[i] == 0)[0][0] + 1 for i in range(len(order))])

    report = {
        "data": str(args.data),
        "groups": str(args.groups),
        "checkpoint": str(args.checkpoint),
        "mode": mode,
        "test_groups": int(len(test_groups)),
        "top1": float(top1.mean()),
        "mean_rank": float(ranks.mean()),
        "pairwise_rows": rows,
    }

    args.out.write_text(json.dumps(report, indent=2) + "\n")

    md = args.out.with_suffix(".md")
    lines = [
        f"# Mixed hard Transformer pairwise audit: {mode}",
        "",
        f"- data: `{args.data}`",
        f"- groups: `{args.groups}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- mode: `{mode}`",
        f"- test groups: `{len(test_groups)}`",
        f"- top-1: `{top1.mean():.6f}`",
        f"- mean rank: `{ranks.mean():.6f}`",
        "",
        "| negative type | pairwise accuracy | mean margin | count |",
        "| --- | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['negative_type']} | {r['pairwise_accuracy']:.6f} | "
            f"{r['mean_margin']:.6f} | {r['count']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "`same_state_diff_action` tests whether the model uses action information.",
        "`same_action_diff_state` tests whether the model uses state-dependent visual dynamics.",
    ]

    md.write_text("\n".join(lines) + "\n")
    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, din, dout, hidden, layers, dropout):
        super().__init__()
        mods = []
        d = din
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        mods.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-groups", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def cond(z, a, mode):
    if mode == "full":
        return torch.cat([z, a], -1)
    if mode == "action_only":
        return a
    if mode == "state_only":
        return z
    if mode == "no_context":
        return torch.ones((z.shape[0], 1), device=z.device, dtype=z.dtype)
    raise ValueError(mode)


@torch.no_grad()
def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.out.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    g = np.load(args.groups, allow_pickle=True)

    z = d["z_current"].astype(np.float32).reshape(len(d["z_current"]), -1)
    y = d["z_future"].astype(np.float32).reshape(len(d["z_future"]), -1)
    a = d["action"].astype(np.float32)
    delta = y - z

    anchor = g["anchor_indices"].astype(np.int64)
    cand = g["candidate_indices"].astype(np.int64)
    neg_type = g["negative_type"].astype(str)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]
    mode = ckpt_args["mode"]

    model = MLP(
        int(ckpt["input_dim"]),
        int(ckpt["output_dim"]),
        int(ckpt_args["hidden"]),
        int(ckpt_args["layers"]),
        float(ckpt_args["dropout"]),
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_groups = cand.shape[0]
    rng = np.random.default_rng(int(ckpt_args.get("seed", 0)))
    perm = rng.permutation(n_groups)

    n_train = int(float(ckpt_args.get("train_frac", 0.8)) * n_groups)
    n_val = int(float(ckpt_args.get("val_frac", 0.1)) * n_groups)
    test_groups = perm[n_train + n_val:]

    all_rows = []

    preds = []
    cands = []
    types = []

    for s in range(0, len(test_groups), args.batch_groups):
        gids = test_groups[s:s + args.batch_groups]
        anch = anchor[gids]
        c = cand[gids]

        zc = torch.from_numpy(z[anch]).float().to(args.device)
        aa = torch.from_numpy(a[anch]).float().to(args.device)

        pred = model(cond(zc, aa, mode)).cpu().numpy()

        preds.append(pred)
        cands.append(delta[c])
        types.append(neg_type[gids])

    pred = np.concatenate(preds, axis=0)
    cand_delta = np.concatenate(cands, axis=0)
    typ = np.concatenate(types, axis=0)

    dist = ((pred[:, None, :] - cand_delta) ** 2).mean(axis=-1)
    pos = dist[:, 0]

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

        all_rows.append({
            "negative_type": t,
            "pairwise_accuracy": float(wins.mean()),
            "mean_margin": float(margins.mean()),
            "count": int(len(wins)),
        })

    # Full top-1 too.
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
        "pairwise_rows": all_rows,
    }

    args.out.write_text(json.dumps(report, indent=2) + "\n")

    md = args.out.with_suffix(".md")
    lines = [
        f"# Mixed hard pairwise audit: {mode}",
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

    for r in all_rows:
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

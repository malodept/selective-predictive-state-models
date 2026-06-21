from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class DeltaTransformer(nn.Module):
    def __init__(
        self,
        token_dim: int,
        action_dim: int,
        tokens: int,
        model_dim: int,
        heads: int,
        layers: int,
        dropout: float,
        mode: str,
    ):
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

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, z_tokens: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # z_tokens: [B,T,D], action: [B,A]
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
            a = self.action_proj(action).unsqueeze(1)
            h = h + a

        h = h + self.pos
        h = self.encoder(h)
        h = self.norm(h)
        return self.out_proj(h)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--mode", choices=["full", "action_only", "state_only", "no_context"], required=True)

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-groups", type=int, default=64)
    p.add_argument("--eval-batch-groups", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--model-dim", type=int, default=384)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--lambda-mse", type=float, default=0.01)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def rank_metrics(pred_delta, cand_delta):
    # pred_delta: [G,T,D], cand_delta: [G,K,T,D]
    dist = ((pred_delta[:, None] - cand_delta) ** 2).mean(axis=(2, 3))
    order = np.argsort(dist, axis=1)
    top1 = order[:, 0] == 0
    ranks = np.array([np.where(order[i] == 0)[0][0] + 1 for i in range(len(order))])
    margin = dist[:, 1:].min(axis=1) - dist[:, 0]
    return {
        "top1": float(top1.mean()),
        "mean_rank": float(ranks.mean()),
        "positive_margin_frac": float((margin > 0).mean()),
        "mean_margin": float(margin.mean()),
    }


@torch.no_grad()
def evaluate(model, z, y, a, anchor, cand, gids, variant, batch_groups, device, seed):
    model.eval()
    rng = np.random.default_rng(seed)
    delta = y - z

    preds = []
    tgts = []

    for s in range(0, len(gids), batch_groups):
        ids = gids[s:s + batch_groups]
        anch = anchor[ids]
        c = cand[ids]

        zc = z[anch].copy()
        aa = a[anch].copy()

        if variant == "original":
            pass
        elif variant == "action_zero":
            aa = np.zeros_like(aa)
        elif variant == "cond_state_shuffle":
            zc = zc[rng.permutation(len(zc))]
        elif variant == "cond_state_zero":
            zc = np.zeros_like(zc)
        else:
            raise ValueError(variant)

        zt = torch.from_numpy(zc).float().to(device)
        at = torch.from_numpy(aa).float().to(device)

        pred = model(zt, at)
        preds.append(pred.cpu().numpy())
        tgts.append(delta[c])

    return rank_metrics(np.concatenate(preds), np.concatenate(tgts))


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    g = np.load(args.groups, allow_pickle=True)

    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    a = d["action"].astype(np.float32)

    anchor = g["anchor_indices"].astype(np.int64)
    cand = g["candidate_indices"].astype(np.int64)

    n_groups, K = cand.shape
    tokens, token_dim = z.shape[1], z.shape[2]
    action_dim = a.shape[1]

    if "split" in g.files:
        split = g["split"].astype(str)
        train_groups = np.where(split == "train")[0]
        val_groups = np.where(split == "val")[0]
        test_groups = np.where(split == "test")[0]
        print(
            f"using explicit split from groups file: "
            f"train={len(train_groups)} val={len(val_groups)} test={len(test_groups)}",
            flush=True,
        )
    else:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(n_groups)

        n_train = int(args.train_frac * n_groups)
        n_val = int(args.val_frac * n_groups)

        train_groups = perm[:n_train]
        val_groups = perm[n_train:n_train + n_val]
        test_groups = perm[n_train + n_val:]

    model = DeltaTransformer(
        token_dim=token_dim,
        action_dim=action_dim,
        tokens=tokens,
        model_dim=args.model_dim,
        heads=args.heads,
        layers=args.layers,
        dropout=args.dropout,
        mode=args.mode,
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

    delta = y - z

    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_groups).long()),
        batch_size=args.batch_groups,
        shuffle=True,
    )

    best_val = -1.0
    best_state = None
    train_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for (gid_t,) in loader:
            gids = gid_t.numpy()
            anch = anchor[gids]
            c = cand[gids]

            zc = torch.from_numpy(z[anch]).float().to(args.device)
            aa = torch.from_numpy(a[anch]).float().to(args.device)
            tgt = torch.from_numpy(delta[c]).float().to(args.device)

            pred = model(zc, aa)
            dist = ((pred[:, None] - tgt) ** 2).mean(dim=(2, 3))
            logits = -dist / args.temperature

            loss_ce = ce(logits, torch.zeros(len(gids), dtype=torch.long, device=args.device))
            loss_mse = ((pred - tgt[:, 0]) ** 2).mean()
            loss = loss_ce + args.lambda_mse * loss_mse

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))

        val_m = evaluate(
            model, z, y, a, anchor, cand, val_groups,
            "original", args.eval_batch_groups, args.device, args.seed,
        )

        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_top1": val_m["top1"],
            "val_rank": val_m["mean_rank"],
        }
        train_rows.append(row)

        if val_m["top1"] > best_val:
            best_val = val_m["top1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"mode={args.mode} epoch={epoch:03d} loss={row['loss']:.6f} "
                f"val_top1={row['val_top1']:.6f} val_rank={row['val_rank']:.3f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "best_val_top1": float(best_val),
            "tokens": int(tokens),
            "token_dim": int(token_dim),
            "action_dim": int(action_dim),
        },
        args.out_dir / "checkpoint.pt",
    )

    eval_rows = []
    for split, gids in [("train", train_groups), ("val", val_groups), ("test", test_groups)]:
        for variant in ["original", "action_zero", "cond_state_shuffle", "cond_state_zero"]:
            m = evaluate(
                model, z, y, a, anchor, cand, gids,
                variant, args.eval_batch_groups, args.device, args.seed,
            )
            m.update({"split": split, "variant": variant})
            eval_rows.append(m)

    report = {
        "data": str(args.data),
        "groups": str(args.groups),
        "mode": args.mode,
        "n_groups": int(n_groups),
        "candidates": int(K),
        "tokens": int(tokens),
        "token_dim": int(token_dim),
        "train_val_test": [int(len(train_groups)), int(len(val_groups)), int(len(test_groups))],
        "chance_top1": float(1.0 / K),
        "best_val_top1": float(best_val),
        "training_rows": train_rows,
        "eval_rows": eval_rows,
    }

    args.report.write_text(json.dumps(report, indent=2) + "\n")

    md = args.report.with_suffix(".md")
    lines = [
        f"# Mixed hard delta Transformer: {args.mode}",
        "",
        f"- data: `{args.data}`",
        f"- groups: `{args.groups}`",
        f"- mode: `{args.mode}`",
        f"- groups count: `{n_groups}`",
        f"- candidates: `{K}`",
        f"- tokens: `{tokens}`",
        f"- token dim: `{token_dim}`",
        f"- model dim: `{args.model_dim}`",
        f"- layers: `{args.layers}`",
        f"- heads: `{args.heads}`",
        f"- train/val/test groups: `{len(train_groups)}/{len(val_groups)}/{len(test_groups)}`",
        f"- chance top-1: `{1.0 / K:.6f}`",
        f"- best val top-1: `{best_val:.6f}`",
        "",
        "| split | variant | top-1 | mean rank | positive margin frac | mean margin |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for r in eval_rows:
        lines.append(
            f"| {r['split']} | {r['variant']} | {r['top1']:.6f} | {r['mean_rank']:.6f} | "
            f"{r['positive_margin_frac']:.6f} | {r['mean_margin']:.6f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.",
        "It is evaluated on mixed hard negatives in delta space.",
    ]

    md.write_text("\n".join(lines) + "\n")
    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()

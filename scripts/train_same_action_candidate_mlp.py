from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        mods = []
        d = input_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        mods.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


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
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--lambda-mse", type=float, default=0.01)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def condition(z: torch.Tensor, a: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "full":
        return torch.cat([z, a], dim=-1)
    if mode == "action_only":
        return a
    if mode == "state_only":
        return z
    if mode == "no_context":
        return torch.ones((z.shape[0], 1), device=z.device, dtype=z.dtype)
    raise ValueError(mode)


def metrics(pred: np.ndarray, ycand: np.ndarray) -> dict:
    # pred: [G,D], ycand: [G,K,D]
    dist = ((pred[:, None, :] - ycand) ** 2).mean(axis=-1)
    rank = np.argsort(dist, axis=1)
    top1 = rank[:, 0] == 0
    true_rank = np.array([np.where(rank[i] == 0)[0][0] + 1 for i in range(len(rank))])
    margin = dist[:, 1:].min(axis=1) - dist[:, 0]
    return {
        "top1": float(top1.mean()),
        "mean_rank": float(true_rank.mean()),
        "positive_margin_frac": float((margin > 0).mean()),
        "mean_margin": float(margin.mean()),
    }


@torch.no_grad()
def evaluate(model, z, y, a, anchor, cand, group_ids, mode, variant, batch_groups, device, seed):
    model.eval()
    rng = np.random.default_rng(seed)
    preds, ys = [], []

    for s in range(0, len(group_ids), batch_groups):
        gids = group_ids[s:s + batch_groups]
        anch = anchor[gids]
        c = cand[gids]
        G, K = c.shape

        z_base_np = z[anch]
        z_cond_np = z_base_np.copy()
        a_np = a[anch].copy()

        if variant == "original":
            pass
        elif variant == "action_zero":
            a_np = np.zeros_like(a_np)
        elif variant == "cond_state_shuffle":
            z_cond_np = z_cond_np[rng.permutation(G)]
        elif variant == "cond_state_zero":
            z_cond_np = np.zeros_like(z_cond_np)
        else:
            raise ValueError(variant)

        z_base = torch.from_numpy(z_base_np).float().to(device)
        z_cond = torch.from_numpy(z_cond_np).float().to(device)
        aa = torch.from_numpy(a_np).float().to(device)

        pred = z_base + model(condition(z_cond, aa, mode))
        preds.append(pred.cpu().numpy())
        ys.append(y[c])

    return metrics(np.concatenate(preds), np.concatenate(ys))


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

    z = d["z_current"].astype(np.float32).reshape(len(d["z_current"]), -1)
    y = d["z_future"].astype(np.float32).reshape(len(d["z_future"]), -1)
    a = d["action"].astype(np.float32)

    anchor = g["anchor_indices"].astype(np.int64)
    cand = g["candidate_indices"].astype(np.int64)

    n_groups, K = cand.shape
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_groups)

    n_train = int(args.train_frac * n_groups)
    n_val = int(args.val_frac * n_groups)

    train_groups = perm[:n_train]
    val_groups = perm[n_train:n_train + n_val]
    test_groups = perm[n_train + n_val:]

    z_sample = torch.zeros((1, z.shape[1]))
    a_sample = torch.zeros((1, a.shape[1]))
    input_dim = condition(z_sample, a_sample, args.mode).shape[1]

    model = MLP(input_dim, z.shape[1], args.hidden, args.layers, args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce = nn.CrossEntropyLoss()

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
            G, K = c.shape

            z_base_np = z[anch]
            y_cand_np = y[c]
            a_np = a[anch]

            z_base = torch.from_numpy(z_base_np).float().to(args.device)
            z_cond = z_base
            aa = torch.from_numpy(a_np).float().to(args.device)
            y_cand = torch.from_numpy(y_cand_np).float().to(args.device)

            pred = z_base + model(condition(z_cond, aa, args.mode))
            dist = ((pred[:, None, :] - y_cand) ** 2).mean(dim=-1)
            logits = -dist / args.temperature

            loss_ce = ce(logits, torch.zeros(G, dtype=torch.long, device=args.device))
            loss_mse = ((pred - y_cand[:, 0, :]) ** 2).mean()
            loss = loss_ce + args.lambda_mse * loss_mse

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        val_m = evaluate(model, z, y, a, anchor, cand, val_groups, args.mode, "original", args.eval_batch_groups, args.device, args.seed)

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
            "input_dim": int(input_dim),
            "output_dim": int(z.shape[1]),
            "best_val_top1": float(best_val),
        },
        args.out_dir / "checkpoint.pt",
    )

    eval_rows = []
    for split, gids in [("train", train_groups), ("val", val_groups), ("test", test_groups)]:
        for variant in ["original", "action_zero", "cond_state_shuffle", "cond_state_zero"]:
            m = evaluate(model, z, y, a, anchor, cand, gids, args.mode, variant, args.eval_batch_groups, args.device, args.seed)
            m.update({"split": split, "variant": variant})
            eval_rows.append(m)

    report = {
        "data": str(args.data),
        "groups": str(args.groups),
        "mode": args.mode,
        "n_groups": int(n_groups),
        "candidates": int(K),
        "train_val_test": [int(len(train_groups)), int(len(val_groups)), int(len(test_groups))],
        "chance_top1": float(1.0 / K),
        "best_val_top1": float(best_val),
        "training_rows": train_rows,
        "eval_rows": eval_rows,
    }

    args.report.write_text(json.dumps(report, indent=2) + "\n")

    md = args.report.with_suffix(".md")
    lines = [
        f"# Same-action hard-negative candidate MLP: {args.mode}",
        "",
        f"- data: `{args.data}`",
        f"- groups: `{args.groups}`",
        f"- mode: `{args.mode}`",
        f"- groups count: `{n_groups}`",
        f"- candidates: `{K}`",
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

    md.write_text("\n".join(lines) + "\n")
    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()

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


def build_condition(z_flat: torch.Tensor, action: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "full":
        return torch.cat([z_flat, action], dim=-1)
    if mode == "action_only":
        return action
    if mode == "state_only":
        return z_flat
    if mode == "no_context":
        return torch.ones((z_flat.shape[0], 1), device=z_flat.device, dtype=z_flat.dtype)
    raise ValueError(mode)


def rank_metrics(pred: np.ndarray, futures: np.ndarray):
    # pred: [G, K, D], futures: [G, K, D]
    mse = ((pred[:, :, None] - futures[:, None, :]) ** 2).mean(axis=3)

    top1, ranks, margins = [], [], []

    for g in range(mse.shape[0]):
        for k in range(mse.shape[1]):
            order = np.argsort(mse[g, k])
            rank = int(np.where(order == k)[0][0]) + 1
            ranks.append(rank)
            top1.append(rank == 1)

            off = np.delete(mse[g, k], k)
            margins.append(float(off.min() - mse[g, k, k]))

    margins = np.asarray(margins)
    return {
        "top1": float(np.mean(top1)),
        "mean_rank": float(np.mean(ranks)),
        "positive_margin_frac": float(np.mean(margins > 0)),
        "mean_margin": float(np.mean(margins)),
    }


def make_action_intervention(action_cand: np.ndarray, intervention: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = action_cand.copy()
    G, K, _ = out.shape

    if intervention == "original":
        return out
    if intervention == "zero":
        return np.zeros_like(out)
    if intervention == "within_group_shuffle":
        for g in range(G):
            out[g] = out[g, rng.permutation(K)]
        return out
    if intervention == "within_group_reverse":
        return out[:, ::-1].copy()

    raise ValueError(intervention)


@torch.no_grad()
def predict_groups(model, z_flat_np, y_flat_np, action_np, cand_np, groups_np, mode, intervention, batch_groups, device, seed):
    model.eval()

    all_pred = []
    all_future = []

    for start in range(0, len(groups_np), batch_groups):
        group_ids = groups_np[start:start + batch_groups]
        cand = cand_np[group_ids]
        G, K = cand.shape

        z_anchor = z_flat_np[cand[:, 0]]
        y_cand = y_flat_np[cand]
        action_cand = action_np[cand]
        action_eval = make_action_intervention(action_cand, intervention, seed + start)

        z_rep_np = np.repeat(z_anchor[:, None, :], K, axis=1).reshape(G * K, -1)
        a_rep_np = action_eval.reshape(G * K, -1)

        z_rep = torch.from_numpy(z_rep_np).float().to(device)
        a_rep = torch.from_numpy(a_rep_np).float().to(device)

        cond = build_condition(z_rep, a_rep, mode)
        delta = model(cond)
        pred = z_rep + delta

        all_pred.append(pred.cpu().numpy().reshape(G, K, -1))
        all_future.append(y_cand.reshape(G, K, -1))

    return np.concatenate(all_pred, axis=0), np.concatenate(all_future, axis=0)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)

    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    action = d["action"].astype(np.float32)
    cand = d["candidate_indices"].astype(np.int64)

    z_flat = z.reshape(len(z), -1)
    y_flat = y.reshape(len(y), -1)

    n_groups, K = cand.shape
    rng = np.random.default_rng(args.seed)
    group_perm = rng.permutation(n_groups)

    n_train = int(args.train_frac * n_groups)
    n_val = int(args.val_frac * n_groups)

    train_groups = group_perm[:n_train]
    val_groups = group_perm[n_train:n_train + n_val]
    test_groups = group_perm[n_train + n_val:]

    # Determine input dimension.
    sample_z = torch.zeros((1, z_flat.shape[1]))
    sample_a = torch.zeros((1, action.shape[1]))
    input_dim = build_condition(sample_z, sample_a, args.mode).shape[1]

    model = MLP(
        input_dim=input_dim,
        output_dim=z_flat.shape[1],
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_loss = nn.CrossEntropyLoss()

    train_ds = TensorDataset(torch.from_numpy(train_groups).long())
    train_loader = DataLoader(train_ds, batch_size=args.batch_groups, shuffle=True)

    best_val_top1 = -1.0
    best_state = None
    rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        ce_losses = []
        mse_losses = []

        for (group_ids_t,) in train_loader:
            group_ids = group_ids_t.numpy()
            cand_b = cand[group_ids]
            G, K = cand_b.shape

            z_anchor_np = z_flat[cand_b[:, 0]]
            y_cand_np = y_flat[cand_b]
            a_cand_np = action[cand_b]

            z_rep_np = np.repeat(z_anchor_np[:, None, :], K, axis=1).reshape(G * K, -1)
            a_rep_np = a_cand_np.reshape(G * K, -1)

            z_rep = torch.from_numpy(z_rep_np).float().to(args.device)
            a_rep = torch.from_numpy(a_rep_np).float().to(args.device)
            y_cand = torch.from_numpy(y_cand_np.reshape(G, K, -1)).float().to(args.device)

            cond = build_condition(z_rep, a_rep, args.mode)
            delta = model(cond)
            pred = (z_rep + delta).reshape(G, K, -1)

            # Distance matrix: prediction for action k against every future j.
            dist = ((pred[:, :, None, :] - y_cand[:, None, :, :]) ** 2).mean(dim=-1)
            logits = -dist / args.temperature

            targets = torch.arange(K, device=args.device).repeat(G)
            loss_ce = ce_loss(logits.reshape(G * K, K), targets)

            diag = pred - y_cand
            loss_mse = (diag ** 2).mean()

            loss = loss_ce + args.lambda_mse * loss_mse

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))
            ce_losses.append(float(loss_ce.item()))
            mse_losses.append(float(loss_mse.item()))

        pred_val, fut_val = predict_groups(
            model, z_flat, y_flat, action, cand, val_groups,
            args.mode, "original", args.eval_batch_groups, args.device, args.seed,
        )
        val_metrics = rank_metrics(pred_val, fut_val)

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_ce": float(np.mean(ce_losses)),
            "train_diag_mse": float(np.mean(mse_losses)),
            "val_top1": val_metrics["top1"],
            "val_mean_rank": val_metrics["mean_rank"],
            "val_margin": val_metrics["mean_margin"],
        }
        rows.append(row)

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"mode={args.mode} epoch={epoch:03d} "
                f"loss={row['train_loss']:.6f} ce={row['train_ce']:.6f} "
                f"diag_mse={row['train_diag_mse']:.6f} "
                f"val_top1={row['val_top1']:.6f} val_rank={row['val_mean_rank']:.3f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "input_dim": int(input_dim),
            "output_dim": int(z_flat.shape[1]),
            "best_val_top1": float(best_val_top1),
        },
        args.out_dir / "checkpoint.pt",
    )

    eval_rows = []
    for split_name, groups in [("train", train_groups), ("val", val_groups), ("test", test_groups)]:
        for intervention in ["original", "zero", "within_group_shuffle", "within_group_reverse"]:
            pred, fut = predict_groups(
                model, z_flat, y_flat, action, cand, groups,
                args.mode, intervention, args.eval_batch_groups, args.device, args.seed,
            )
            m = rank_metrics(pred, fut)
            m.update({"split": split_name, "intervention": intervention})
            eval_rows.append(m)

    report = {
        "data": str(args.data),
        "mode": args.mode,
        "out_dir": str(args.out_dir),
        "groups": int(n_groups),
        "candidates": int(K),
        "train_groups": int(len(train_groups)),
        "val_groups": int(len(val_groups)),
        "test_groups": int(len(test_groups)),
        "chance_top1": float(1.0 / K),
        "best_val_top1": float(best_val_top1),
        "training_rows": rows,
        "eval_rows": eval_rows,
    }

    args.report.write_text(json.dumps(report, indent=2) + "\n")

    md = args.report.with_suffix(".md")
    lines = [
        f"# Exact-intervention candidate MLP: {args.mode}",
        "",
        f"- data: `{args.data}`",
        f"- mode: `{args.mode}`",
        f"- groups: `{n_groups}`",
        f"- candidates: `{K}`",
        f"- train/val/test groups: `{len(train_groups)}/{len(val_groups)}/{len(test_groups)}`",
        f"- chance top-1: `{1.0 / K:.6f}`",
        f"- best val top-1: `{best_val_top1:.6f}`",
        f"- temperature: `{args.temperature}`",
        f"- lambda MSE: `{args.lambda_mse}`",
        "",
        "## Candidate-matching evaluation",
        "",
        "| split | intervention | top-1 | mean rank | positive margin frac | mean margin |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for r in eval_rows:
        lines.append(
            f"| {r['split']} | {r['intervention']} | {r['top1']:.6f} | "
            f"{r['mean_rank']:.6f} | {r['positive_margin_frac']:.6f} | {r['mean_margin']:.6f} |"
        )

    md.write_text("\n".join(lines) + "\n")

    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()

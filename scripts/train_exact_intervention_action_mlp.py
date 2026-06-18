from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ResidualActionMLP(nn.Module):
    def __init__(
        self,
        z_dim: int,
        action_dim: int,
        hidden: int = 256,
        layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()

        modules = []
        d = z_dim + action_dim

        for _ in range(layers):
            modules.append(nn.Linear(d, hidden))
            modules.append(nn.GELU())
            modules.append(nn.Dropout(dropout))
            d = hidden

        modules.append(nn.Linear(d, z_dim))
        self.net = nn.Sequential(*modules)

    def forward(self, z_flat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_flat, action], dim=-1))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def rank_metrics(pred: np.ndarray, futures: np.ndarray):
    # pred: [G, K, T, D]
    # futures: [G, K, T, D]
    mse = ((pred[:, :, None] - futures[:, None, :]) ** 2).mean(axis=(3, 4))

    top1 = []
    ranks = []
    margins = []

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


@torch.no_grad()
def predict(model, z_flat, action, batch_size, device):
    model.eval()
    preds = []

    for i in range(0, len(z_flat), batch_size):
        z_b = torch.from_numpy(z_flat[i:i + batch_size]).float().to(device)
        a_b = torch.from_numpy(action[i:i + batch_size]).float().to(device)

        delta = model(z_b, a_b)
        pred = z_b + delta
        preds.append(pred.cpu().numpy())

    return np.concatenate(preds, axis=0)


def eval_groups(
    model,
    z_flat,
    y_tokens,
    action,
    candidate_indices,
    group_indices,
    batch_size,
    device,
    mode,
    seed,
):
    rng = np.random.default_rng(seed)

    cand = candidate_indices[group_indices]
    G, K = cand.shape

    z_anchor = z_flat[cand[:, 0]]
    y_cand = y_tokens[cand]
    action_cand = action[cand]

    if mode == "original":
        action_eval = action_cand.copy()
    elif mode == "zero":
        action_eval = np.zeros_like(action_cand)
    elif mode == "within_group_shuffle":
        action_eval = action_cand.copy()
        for g in range(G):
            action_eval[g] = action_eval[g, rng.permutation(K)]
    elif mode == "within_group_reverse":
        action_eval = action_cand[:, ::-1].copy()
    else:
        raise ValueError(mode)

    z_rep = np.repeat(z_anchor[:, None, :], K, axis=1)

    pred_flat = predict(
        model=model,
        z_flat=z_rep.reshape(G * K, -1),
        action=action_eval.reshape(G * K, -1),
        batch_size=batch_size,
        device=device,
    )

    pred = pred_flat.reshape(G, K, *y_cand.shape[2:])
    return rank_metrics(pred, y_cand)


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)

    z_tokens = d["z_current"].astype(np.float32)
    y_tokens = d["z_future"].astype(np.float32)
    action = d["action"].astype(np.float32)
    candidate_indices = d["candidate_indices"].astype(np.int64)

    n_groups, K = candidate_indices.shape
    tokens, dim = z_tokens.shape[1], z_tokens.shape[2]

    z_flat = z_tokens.reshape(len(z_tokens), -1)
    y_flat = y_tokens.reshape(len(y_tokens), -1)
    delta_flat = y_flat - z_flat

    rng = np.random.default_rng(args.seed)
    group_perm = rng.permutation(n_groups)

    n_train = int(args.train_frac * n_groups)
    n_val = int(args.val_frac * n_groups)

    train_groups = group_perm[:n_train]
    val_groups = group_perm[n_train:n_train + n_val]
    test_groups = group_perm[n_train + n_val:]

    train_idx = candidate_indices[train_groups].reshape(-1)
    val_idx = candidate_indices[val_groups].reshape(-1)

    train_ds = TensorDataset(
        torch.from_numpy(z_flat[train_idx]).float(),
        torch.from_numpy(action[train_idx]).float(),
        torch.from_numpy(delta_flat[train_idx]).float(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = ResidualActionMLP(
        z_dim=z_flat.shape[1],
        action_dim=action.shape[1],
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
    ).to(args.device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    training_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for zb, ab, db in train_loader:
            zb = zb.to(args.device)
            ab = ab.to(args.device)
            db = db.to(args.device)

            pred_delta = model(zb, ab)
            loss = loss_fn(pred_delta, db)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))

        pred_val = predict(
            model=model,
            z_flat=z_flat[val_idx],
            action=action[val_idx],
            batch_size=args.batch_size,
            device=args.device,
        )

        val_future_mse = float(np.mean((pred_val - y_flat[val_idx]) ** 2))
        train_delta_mse = float(np.mean(losses))

        row = {
            "epoch": epoch,
            "train_delta_mse": train_delta_mse,
            "val_future_mse": val_future_mse,
        }
        training_rows.append(row)

        if val_future_mse < best_val:
            best_val = val_future_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} "
                f"train_delta_mse={train_delta_mse:.8f} "
                f"val_future_mse={val_future_mse:.8f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "tokens": int(tokens),
            "dim": int(dim),
            "action_dim": int(action.shape[1]),
            "best_val_future_mse": float(best_val),
        },
        args.out_dir / "checkpoint.pt",
    )

    eval_rows = []

    for split_name, groups in [
        ("train", train_groups),
        ("val", val_groups),
        ("test", test_groups),
    ]:
        for mode in [
            "original",
            "zero",
            "within_group_shuffle",
            "within_group_reverse",
        ]:
            m = eval_groups(
                model=model,
                z_flat=z_flat,
                y_tokens=y_tokens,
                action=action,
                candidate_indices=candidate_indices,
                group_indices=groups,
                batch_size=args.batch_size,
                device=args.device,
                mode=mode,
                seed=args.seed,
            )
            m.update({"split": split_name, "mode": mode})
            eval_rows.append(m)

    report = {
        "data": str(args.data),
        "out_dir": str(args.out_dir),
        "groups": int(n_groups),
        "candidates": int(K),
        "train_groups": int(len(train_groups)),
        "val_groups": int(len(val_groups)),
        "test_groups": int(len(test_groups)),
        "chance_top1": float(1.0 / K),
        "best_val_future_mse": float(best_val),
        "training_rows": training_rows,
        "eval_rows": eval_rows,
    }

    args.report.write_text(json.dumps(report, indent=2) + "\n")

    md = args.report.with_suffix(".md")
    lines = [
        "# Exact-intervention action MLP",
        "",
        f"- data: `{args.data}`",
        f"- groups: `{n_groups}`",
        f"- candidates: `{K}`",
        f"- train/val/test groups: `{len(train_groups)}/{len(val_groups)}/{len(test_groups)}`",
        f"- chance top-1: `{1.0 / K:.6f}`",
        f"- best validation future MSE: `{best_val:.8f}`",
        "",
        "## Candidate-matching evaluation",
        "",
        "| split | mode | top-1 | mean rank | positive margin frac | mean margin |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for r in eval_rows:
        lines.append(
            f"| {r['split']} | {r['mode']} | {r['top1']:.6f} | "
            f"{r['mean_rank']:.6f} | {r['positive_margin_frac']:.6f} | "
            f"{r['mean_margin']:.6f} |"
        )

    lines += [
        "",
        "## Interpretation rule",
        "",
        "A successful exact-intervention learner should have test `original` far above chance and clearly above zero/shuffle interventions.",
    ]

    md.write_text("\n".join(lines) + "\n")

    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()

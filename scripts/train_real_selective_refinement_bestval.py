from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_npz(path: str):
    data = np.load(path, allow_pickle=True)

    if "z_current" not in data or "z_future" not in data:
        raise KeyError(f"{path} must contain z_current and z_future")

    z_current = data["z_current"].astype(np.float32)
    z_future = data["z_future"].astype(np.float32)

    if "action" in data:
        action = data["action"].astype(np.float32)
    else:
        action = np.zeros((z_current.shape[0], 0), dtype=np.float32)

    x = np.concatenate([z_current, action], axis=1).astype(np.float32)
    y = z_future.astype(np.float32)

    return x, y, {
        "n_samples": int(x.shape[0]),
        "input_dim": int(x.shape[1]),
        "latent_dim": int(y.shape[1]),
        "action_dim": int(action.shape[1]),
    }


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, layers: int, dropout: float = 0.0):
        super().__init__()
        blocks = []
        dim = input_dim
        for _ in range(layers):
            blocks.append(nn.Linear(dim, hidden_dim))
            blocks.append(nn.GELU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            dim = hidden_dim
        blocks.append(nn.Linear(dim, output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def make_loader(x, y, batch_size: int, shuffle: bool):
    tx = torch.from_numpy(x)
    ty = torch.from_numpy(y)
    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    outs = []
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)
    for xb in loader:
        xb = xb.to(device)
        outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def mse_loss_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def per_sample_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((pred - target) ** 2, axis=1)


def train_regressor(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    patience: int,
    name: str,
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    loader = make_loader(train_x, train_y, batch_size=batch_size, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_val = math.inf
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

            total += float(loss.detach().cpu()) * xb.shape[0]
            n += xb.shape[0]

        train_loss = total / max(n, 1)
        val_pred = predict(model, val_x, batch_size=batch_size, device=device)
        val_loss = mse_loss_numpy(val_pred, val_y)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(f"{name} epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if patience > 0 and stale >= patience:
            print(f"{name}: early stop at epoch {epoch}, best epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    return history, best_epoch, best_val


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = ranks[labels == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def train_reliability(
    model: nn.Module,
    train_x: np.ndarray,
    train_labels: np.ndarray,
    val_x: np.ndarray,
    val_labels: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    patience: int,
):
    model.to(device)

    tx = torch.from_numpy(train_x)
    ty = torch.from_numpy(train_labels.astype(np.float32)).view(-1, 1)
    loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True, drop_last=False)

    pos = float(train_labels.sum())
    neg = float(len(train_labels) - train_labels.sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state = copy.deepcopy(model.state_dict())
    best_val = math.inf
    best_epoch = 0
    stale = 0
    history = []

    vx = torch.from_numpy(val_x).to(device)
    vy = torch.from_numpy(val_labels.astype(np.float32)).view(-1, 1).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total += float(loss.detach().cpu()) * xb.shape[0]
            n += xb.shape[0]

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(vx), vy).detach().cpu())

        train_loss = total / max(n, 1)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        print(f"reliability epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if patience > 0 and stale >= patience:
            print(f"reliability: early stop at epoch {epoch}, best epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    return history, best_epoch, best_val


@torch.no_grad()
def reliability_scores(model: nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    scores = []
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)
    for xb in loader:
        xb = xb.to(device)
        s = torch.sigmoid(model(xb)).detach().cpu().numpy().reshape(-1)
        scores.append(s)
    return np.concatenate(scores, axis=0)


def selector_rows(
    cheap_pred: np.ndarray,
    expensive_pred: np.ndarray,
    target: np.ndarray,
    scores: np.ndarray,
    thresholds: list[float],
    cheap_compute: float,
    expensive_compute: float,
    lambda_compute: float,
):
    rows = []

    def evaluate(policy: str, selected: np.ndarray, threshold):
        pred = np.where(selected[:, None], expensive_pred, cheap_pred)
        err = float(np.mean(per_sample_mse(pred, target)))
        selected_fraction = float(np.mean(selected))
        compute = float(cheap_compute + selected_fraction * (expensive_compute - cheap_compute))
        utility = float(-err - lambda_compute * compute)
        rows.append({
            "threshold": threshold,
            "mean_error": err,
            "mean_compute": compute,
            "utility": utility,
            "selected_fraction": selected_fraction,
            "policy": policy,
        })

    for tau in thresholds:
        selected = scores >= tau
        evaluate(f"threshold={tau:.2f}", selected, tau)

    evaluate("cheap-only", np.zeros_like(scores, dtype=bool), None)
    evaluate("all-expensive", np.ones_like(scores, dtype=bool), None)

    rows.sort(key=lambda r: r["utility"], reverse=True)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = ["threshold", "mean_error", "mean_compute", "utility", "selected_fraction", "policy"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in keys})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--cheap-hidden", type=int, default=256)
    parser.add_argument("--cheap-layers", type=int, default=2)
    parser.add_argument("--expensive-hidden", type=int, default=512)
    parser.add_argument("--expensive-layers", type=int, default=4)
    parser.add_argument("--reliability-hidden", type=int, default=256)

    parser.add_argument("--cheap-epochs", type=int, default=10)
    parser.add_argument("--expensive-epochs", type=int, default=80)
    parser.add_argument("--reliability-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=12)

    parser.add_argument("--hard-fraction", type=float, default=0.30)
    parser.add_argument("--lambda-compute", type=float, default=0.04)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90])
    parser.add_argument("--cheap-compute", type=float, default=1.0)
    parser.add_argument("--expensive-compute", type=float, default=4.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    set_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_x, train_y, train_info = load_feature_npz(args.train)
    val_x, val_y, val_info = load_feature_npz(args.val)

    input_dim = train_info["input_dim"]
    latent_dim = train_info["latent_dim"]

    cheap = MLP(input_dim, latent_dim, args.cheap_hidden, args.cheap_layers, args.dropout)
    expensive = MLP(input_dim, latent_dim, args.expensive_hidden, args.expensive_layers, args.dropout)
    reliability = MLP(input_dim, 1, args.reliability_hidden, 2, args.dropout)

    cheap_history, cheap_best_epoch, cheap_best_val = train_regressor(
        cheap,
        train_x,
        train_y,
        val_x,
        val_y,
        epochs=args.cheap_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        patience=args.patience,
        name="cheap",
    )

    expensive_history, expensive_best_epoch, expensive_best_val = train_regressor(
        expensive,
        train_x,
        train_y,
        val_x,
        val_y,
        epochs=args.expensive_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        patience=args.patience,
        name="expensive",
    )

    cheap_train_pred = predict(cheap, train_x, args.batch_size, args.device)
    cheap_val_pred = predict(cheap, val_x, args.batch_size, args.device)
    expensive_val_pred = predict(expensive, val_x, args.batch_size, args.device)

    train_cheap_errors = per_sample_mse(cheap_train_pred, train_y)
    val_cheap_errors = per_sample_mse(cheap_val_pred, val_y)

    threshold = float(np.quantile(train_cheap_errors, 1.0 - args.hard_fraction))
    train_labels = (train_cheap_errors >= threshold).astype(np.int64)
    val_labels = (val_cheap_errors >= threshold).astype(np.int64)

    reliability_history, reliability_best_epoch, reliability_best_val = train_reliability(
        reliability,
        train_x,
        train_labels,
        val_x,
        val_labels,
        epochs=args.reliability_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        patience=args.patience,
    )

    scores = reliability_scores(reliability, val_x, args.batch_size, args.device)

    rows = selector_rows(
        cheap_val_pred,
        expensive_val_pred,
        val_y,
        scores,
        args.thresholds,
        args.cheap_compute,
        args.expensive_compute,
        args.lambda_compute,
    )

    cheap_error = mse_loss_numpy(cheap_val_pred, val_y)
    expensive_error = mse_loss_numpy(expensive_val_pred, val_y)
    best = rows[0]

    metrics = {
        "seed": args.seed,
        "train": args.train,
        "val": args.val,
        "data": {
            "train": train_info,
            "val": val_info,
        },
        "model": {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "cheap_hidden": args.cheap_hidden,
            "cheap_layers": args.cheap_layers,
            "expensive_hidden": args.expensive_hidden,
            "expensive_layers": args.expensive_layers,
            "reliability_hidden": args.reliability_hidden,
            "dropout": args.dropout,
        },
        "training": {
            "cheap_epochs_requested": args.cheap_epochs,
            "expensive_epochs_requested": args.expensive_epochs,
            "reliability_epochs_requested": args.reliability_epochs,
            "cheap_best_epoch": cheap_best_epoch,
            "expensive_best_epoch": expensive_best_epoch,
            "reliability_best_epoch": reliability_best_epoch,
            "cheap_best_val": cheap_best_val,
            "expensive_best_val": expensive_best_val,
            "reliability_best_val": reliability_best_val,
            "cheap_history": cheap_history,
            "expensive_history": expensive_history,
            "reliability_history": reliability_history,
        },
        "labeling": {
            "hard_fraction": args.hard_fraction,
            "hard_threshold": threshold,
            "train_positive_rate": float(train_labels.mean()),
            "val_positive_rate": float(val_labels.mean()),
            "train_cheap_error_mean": float(train_cheap_errors.mean()),
            "val_cheap_error_mean": float(val_cheap_errors.mean()),
        },
        "diagnostics": {
            "cheap_error": cheap_error,
            "expensive_error": expensive_error,
            "gap": cheap_error - expensive_error,
            "reliability_auroc_vs_cheap_hard_labels": binary_auc(val_labels, scores),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
        },
        "selector_rows": rows,
        "best_policy": best["policy"],
        "best_error": best["mean_error"],
        "best_compute": best["mean_compute"],
        "best_selected": best["selected_fraction"],
        "best_utility": best["utility"],
    }

    with (out / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    write_csv(out / "selector_utility.csv", rows)

    torch.save(
        {
            "cheap": cheap.state_dict(),
            "expensive": expensive.state_dict(),
            "reliability": reliability.state_dict(),
            "metrics": metrics,
        },
        out / "bestval_checkpoint.pt",
    )

    print(f"Wrote {out / 'metrics.json'}")
    print(f"cheap_error={cheap_error:.6f}")
    print(f"expensive_error={expensive_error:.6f}")
    print(f"gap={cheap_error - expensive_error:.6f}")
    print(
        f"best={best['policy']} error={best['mean_error']:.6f} "
        f"compute={best['mean_compute']:.4f} selected={best['selected_fraction']:.4f} "
        f"utility={best['utility']:.6f}"
    )
    print(
        f"best epochs: cheap={cheap_best_epoch}, "
        f"expensive={expensive_best_epoch}, reliability={reliability_best_epoch}"
    )


if __name__ == "__main__":
    main()

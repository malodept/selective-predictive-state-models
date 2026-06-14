from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_envsplit_residual_predictor import (
    MLP,
    load_npz,
    fit_stats,
    transform_split,
    mean_mse,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


@torch.no_grad()
def predict_delta(
    model: nn.Module,
    x: np.ndarray,
    stats: dict[str, np.ndarray],
    batch_size: int,
    device: str,
) -> np.ndarray:
    model.eval()
    outs = []
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)

    delta_std = torch.from_numpy(stats["delta_std"]).to(device)
    delta_mean = torch.from_numpy(stats["delta_mean"]).to(device)

    for xb in loader:
        xb = xb.to(device)
        pred_norm = model(xb)
        pred_delta = pred_norm * delta_std[None, :] + delta_mean[None, :]
        outs.append(pred_delta.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(outs, axis=0)


def best_alpha(z0: np.ndarray, y: np.ndarray, delta: np.ndarray) -> float:
    true_delta = y - z0
    num = float(np.sum(delta * true_delta))
    den = float(np.sum(delta * delta)) + 1e-12
    return float(np.clip(num / den, 0.0, 1.0))


def geometry(z0: np.ndarray, y: np.ndarray, delta: np.ndarray) -> dict[str, float]:
    true_delta = y - z0

    true_norm = np.linalg.norm(true_delta, axis=1)
    pred_norm = np.linalg.norm(delta, axis=1)

    dot = np.sum(delta * true_delta, axis=1)
    cos = dot / (pred_norm * true_norm + 1e-12)

    return {
        "cosine_mean": float(np.mean(cos)),
        "cosine_median": float(np.median(cos)),
        "cosine_positive_frac": float(np.mean(cos > 0.0)),
        "true_delta_norm_median": float(np.median(true_norm)),
        "pred_delta_norm_median": float(np.median(pred_norm)),
    }


def eval_split(
    model: nn.Module,
    data: dict[str, np.ndarray],
    x: np.ndarray,
    stats: dict[str, np.ndarray],
    batch_size: int,
    device: str,
) -> dict[str, float]:
    z0 = data["z_current"].astype(np.float32)
    y = data["z_future"].astype(np.float32)

    delta = predict_delta(model, x, stats, batch_size, device)

    identity_error = mean_mse(z0, y)
    raw_error = mean_mse(z0 + delta, y)

    alpha = best_alpha(z0, y, delta)
    global_error = mean_mse(z0 + alpha * delta, y)

    g = geometry(z0, y, delta)

    return {
        "identity_error": identity_error,
        "raw_error": raw_error,
        "global_alpha": alpha,
        "global_error": global_error,
        "raw_improvement_vs_identity": identity_error - raw_error,
        "global_improvement_vs_identity": identity_error - global_error,
        **g,
    }


def train_directional(
    model: nn.Module,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_data: dict[str, np.ndarray],
    val_x: np.ndarray,
    stats: dict[str, np.ndarray],
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    lambda_cos: float,
    lambda_norm: float,
    patience: int,
    device: str,
) -> tuple[nn.Module, list[dict[str, float]], dict[str, float]]:
    model.to(device)

    loader = make_loader(train_x, train_y, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    delta_std = torch.from_numpy(stats["delta_std"]).to(device)
    delta_mean = torch.from_numpy(stats["delta_mean"]).to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_eval = None
    best_val_error = float("inf")
    stale = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        total = 0.0
        total_mse = 0.0
        total_cos = 0.0
        total_norm = 0.0
        n = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred_norm = model(xb)

            pred_delta = pred_norm * delta_std[None, :] + delta_mean[None, :]
            true_delta = yb * delta_std[None, :] + delta_mean[None, :]

            mse_loss = F.mse_loss(pred_norm, yb)

            cos = F.cosine_similarity(pred_delta, true_delta, dim=1, eps=1e-8)
            cos_loss = (1.0 - cos).mean()

            pred_lognorm = torch.log(torch.linalg.norm(pred_delta, dim=1) + 1e-6)
            true_lognorm = torch.log(torch.linalg.norm(true_delta, dim=1) + 1e-6)
            norm_loss = F.mse_loss(pred_lognorm, true_lognorm)

            loss = mse_loss + lambda_cos * cos_loss + lambda_norm * norm_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = xb.shape[0]
            total += float(loss.detach().cpu()) * bs
            total_mse += float(mse_loss.detach().cpu()) * bs
            total_cos += float(cos_loss.detach().cpu()) * bs
            total_norm += float(norm_loss.detach().cpu()) * bs
            n += bs

        val_eval = eval_split(model, val_data, val_x, stats, batch_size, device)

        row = {
            "epoch": epoch,
            "train_loss": total / n,
            "train_mse": total_mse / n,
            "train_cos_loss": total_cos / n,
            "train_norm_loss": total_norm / n,
            "val_global_error": val_eval["global_error"],
            "val_global_improvement_vs_identity": val_eval["global_improvement_vs_identity"],
            "val_cosine_mean": val_eval["cosine_mean"],
            "val_cosine_positive_frac": val_eval["cosine_positive_frac"],
            "val_global_alpha": val_eval["global_alpha"],
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"loss={row['train_loss']:.6f} "
            f"mse={row['train_mse']:.6f} "
            f"cos={row['train_cos_loss']:.6f} "
            f"norm={row['train_norm_loss']:.6f} "
            f"val_global={row['val_global_error']:.6f} "
            f"val_dI={row['val_global_improvement_vs_identity']:+.6f} "
            f"val_cos={row['val_cosine_mean']:.6f} "
            f"alpha={row['val_global_alpha']:.4f}"
        )

        if val_eval["global_error"] < best_val_error:
            best_val_error = val_eval["global_error"]
            best_state = copy.deepcopy(model.state_dict())
            best_eval = val_eval
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            print(f"early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    return model, history, best_eval


def write_md(path: Path, metrics: dict) -> None:
    lines = [
        "# Directional residual predictor diagnostic",
        "",
        f"Loss: `MSE + {metrics['lambda_cos']} * cosine_loss + {metrics['lambda_norm']} * lognorm_loss`.",
        "",
        "| split | identity error | raw error | global alpha | global error | global improvement vs identity | cosine mean | cosine positive frac | true Δ norm med | pred Δ norm med |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for split in ["val", "test"]:
        r = metrics[split]
        lines.append(
            f"| {split} | "
            f"{r['identity_error']:.6f} | {r['raw_error']:.6f} | {r['global_alpha']:.6f} | "
            f"{r['global_error']:.6f} | {r['global_improvement_vs_identity']:.6f} | "
            f"{r['cosine_mean']:.6f} | {r['cosine_positive_frac']:.6f} | "
            f"{r['true_delta_norm_median']:.6f} | {r['pred_delta_norm_median']:.6f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-cos", type=float, default=0.05)
    p.add_argument("--lambda-norm", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    set_seed(args.seed)

    train = load_npz(args.train)
    val = load_npz(args.val)
    test = load_npz(args.test)

    stats = fit_stats(train, center_target=False)

    train_x, train_y = transform_split(train, stats)
    val_x, _ = transform_split(val, stats)
    test_x, _ = transform_split(test, stats)

    model = MLP(
        input_dim=train_x.shape[1],
        output_dim=train_y.shape[1],
        hidden_dim=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        zero_init_final=True,
    )

    model, history, best_val = train_directional(
        model,
        train_x,
        train_y,
        val,
        val_x,
        stats,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_cos=args.lambda_cos,
        lambda_norm=args.lambda_norm,
        patience=args.patience,
        device=args.device,
    )

    val_eval = eval_split(model, val, val_x, stats, args.batch_size, args.device)
    test_eval = eval_split(model, test, test_x, stats, args.batch_size, args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "seed": args.seed,
        "lambda_cos": args.lambda_cos,
        "lambda_norm": args.lambda_norm,
        "model": {
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
        },
        "val": val_eval,
        "test": test_eval,
        "history": history,
    }

    with (args.out_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(
        {
            "model": model.state_dict(),
            "stats": {k: torch.from_numpy(v) for k, v in stats.items()},
            "metrics": metrics,
        },
        args.out_dir / "checkpoint.pt",
    )

    write_md(args.out_dir / "directional_residual_diagnostic.md", metrics)

    print(args.out_dir / "directional_residual_diagnostic.md")
    print((args.out_dir / "directional_residual_diagnostic.md").read_text())


if __name__ == "__main__":
    main()

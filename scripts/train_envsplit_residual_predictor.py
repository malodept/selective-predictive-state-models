from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from collections import defaultdict
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


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float = 0.0,
        zero_init_final: bool = True,
    ) -> None:
        super().__init__()

        blocks = []
        dim = input_dim

        for _ in range(layers):
            blocks.append(nn.Linear(dim, hidden_dim))
            blocks.append(nn.GELU())
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            dim = hidden_dim

        final = nn.Linear(dim, output_dim)
        if zero_init_final:
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

        blocks.append(final)
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)

    required = ["z_current", "z_future", "action"]
    for k in required:
        if k not in d.files:
            raise KeyError(f"{path} missing key: {k}")

    out = {k: d[k] for k in d.files}
    out["z_current"] = out["z_current"].astype(np.float32)
    out["z_future"] = out["z_future"].astype(np.float32)
    out["action"] = out["action"].astype(np.float32)

    return out


def feature_matrix(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([data["z_current"], data["action"]], axis=1).astype(np.float32)


def target_delta(data: dict[str, np.ndarray]) -> np.ndarray:
    return (data["z_future"] - data["z_current"]).astype(np.float32)


def fit_stats(train: dict[str, np.ndarray], eps: float = 1e-6, center_target: bool = False) -> dict[str, np.ndarray]:
    x = feature_matrix(train)
    delta = target_delta(train)

    x_mean = x.mean(axis=0).astype(np.float32)
    x_std = x.std(axis=0).astype(np.float32)
    x_std = np.maximum(x_std, eps).astype(np.float32)

    if center_target:
        delta_mean = delta.mean(axis=0).astype(np.float32)
    else:
        # Important: with zero-initialized final layer, output=0 gives identity.
        delta_mean = np.zeros(delta.shape[1], dtype=np.float32)

    delta_std = delta.std(axis=0).astype(np.float32)
    delta_std = np.maximum(delta_std, eps).astype(np.float32)

    # Separate baseline: mean residual actually observed in train.
    train_mean_delta = delta.mean(axis=0).astype(np.float32)

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "train_mean_delta": train_mean_delta,
    }


def transform_split(data: dict[str, np.ndarray], stats: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    x = feature_matrix(data)
    delta = target_delta(data)

    x_norm = (x - stats["x_mean"]) / stats["x_std"]
    y_norm = (delta - stats["delta_mean"]) / stats["delta_std"]

    return x_norm.astype(np.float32), y_norm.astype(np.float32)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


@torch.no_grad()
def predict_delta_norm(model: nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    outs = []

    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)

    for xb in loader:
        xb = xb.to(device)
        pred = model(xb).detach().cpu().numpy()
        outs.append(pred)

    return np.concatenate(outs, axis=0).astype(np.float32)


def reconstruct_prediction(
    data: dict[str, np.ndarray],
    pred_delta_norm: np.ndarray,
    stats: dict[str, np.ndarray],
) -> np.ndarray:
    delta = pred_delta_norm * stats["delta_std"][None, :] + stats["delta_mean"][None, :]
    return (data["z_current"] + delta).astype(np.float32)


def per_sample_mse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((pred - target) ** 2, axis=1)


def mean_mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def baseline_predictions(data: dict[str, np.ndarray], stats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    z0 = data["z_current"]
    mean_delta = stats["train_mean_delta"][None, :]

    return {
        "identity": z0,
        "train_mean_delta": (z0 + mean_delta).astype(np.float32),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data: dict[str, np.ndarray],
    x_norm: np.ndarray,
    stats: dict[str, np.ndarray],
    batch_size: int,
    device: str,
) -> tuple[float, np.ndarray]:
    pred_delta = predict_delta_norm(model, x_norm, batch_size, device)
    pred = reconstruct_prediction(data, pred_delta, stats)
    err = mean_mse(pred, data["z_future"])
    return err, pred


def train_residual_regressor(
    model: nn.Module,
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    stats: dict[str, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    device: str,
    patience: int,
    name: str,
) -> tuple[list[dict[str, float]], int, float, float]:
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    loader = make_loader(train_x, train_y, batch_size=batch_size, shuffle=True)

    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_original = math.inf
    best_val_norm = math.inf
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
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            total += float(loss.detach().cpu()) * xb.shape[0]
            n += xb.shape[0]

        train_norm_loss = total / max(n, 1)

        val_delta_norm = predict_delta_norm(model, val_x, batch_size, device)
        val_norm_loss = float(np.mean((val_delta_norm - val_y) ** 2))
        val_pred = reconstruct_prediction(val_data, val_delta_norm, stats)
        val_original_error = mean_mse(val_pred, val_data["z_future"])

        history.append(
            {
                "epoch": epoch,
                "train_norm_loss": train_norm_loss,
                "val_norm_loss": val_norm_loss,
                "val_original_error": val_original_error,
            }
        )

        print(
            f"{name} epoch={epoch:03d} "
            f"train_norm={train_norm_loss:.6f} "
            f"val_norm={val_norm_loss:.6f} "
            f"val_original={val_original_error:.6f}"
        )

        # Early stopping on the actual metric used for scientific comparison.
        if val_original_error < best_val_original:
            best_val_original = val_original_error
            best_val_norm = val_norm_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if patience > 0 and stale >= patience:
            print(f"{name}: early stop at epoch {epoch}, best epoch {best_epoch}")
            break

    model.load_state_dict(best_state)
    return history, best_epoch, best_val_original, best_val_norm


def group_metrics(
    split_name: str,
    data: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []

    target = data["z_future"]
    env = data["environment"].astype(str) if "environment" in data else np.asarray(["unknown"] * len(target))
    traj = data["trajectory_id"].astype(str) if "trajectory_id" in data else np.asarray(["unknown"] * len(target))

    def add(group: str, mask: np.ndarray) -> None:
        identity_error = mean_mse(predictions["identity"][mask], target[mask])

        for method, pred in predictions.items():
            err = mean_mse(pred[mask], target[mask])
            rows.append(
                {
                    "split": split_name,
                    "group": group,
                    "method": method,
                    "samples": int(mask.sum()),
                    "trajectories": len(set(traj[mask])),
                    "error": err,
                    "identity_error": identity_error,
                    "improvement_vs_identity": identity_error - err,
                    "ratio_vs_identity": err / identity_error if identity_error > 0 else float("nan"),
                }
            )

    add("ALL", np.ones(len(target), dtype=bool))

    for e in sorted(set(env)):
        add(e, env == e)

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return

    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Residual normalized predictor diagnostics",
        "",
        "| split | group | method | samples | trajectories | error | identity error | improvement vs identity | ratio vs identity |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['split']} | {r['group']} | {r['method']} | "
            f"{r['samples']} | {r['trajectories']} | "
            f"{r['error']:.4f} | {r['identity_error']:.4f} | "
            f"{r['improvement_vs_identity']:.4f} | {r['ratio_vs_identity']:.4f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", default=None)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--cheap-hidden", type=int, default=512)
    p.add_argument("--cheap-layers", type=int, default=3)
    p.add_argument("--expensive-hidden", type=int, default=1024)
    p.add_argument("--expensive-layers", type=int, default=5)

    p.add_argument("--cheap-epochs", type=int, default=80)
    p.add_argument("--expensive-epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--center-target", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)

    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    set_seed(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = load_npz(Path(args.train))
    val = load_npz(Path(args.val))
    test = load_npz(Path(args.test)) if args.test else None

    stats = fit_stats(train, center_target=args.center_target)

    train_x, train_y = transform_split(train, stats)
    val_x, val_y = transform_split(val, stats)
    test_x, test_y = transform_split(test, stats) if test is not None else (None, None)

    input_dim = train_x.shape[1]
    latent_dim = train_y.shape[1]

    cheap = MLP(
        input_dim=input_dim,
        output_dim=latent_dim,
        hidden_dim=args.cheap_hidden,
        layers=args.cheap_layers,
        dropout=args.dropout,
        zero_init_final=True,
    )

    expensive = MLP(
        input_dim=input_dim,
        output_dim=latent_dim,
        hidden_dim=args.expensive_hidden,
        layers=args.expensive_layers,
        dropout=args.dropout,
        zero_init_final=True,
    )

    cheap_history, cheap_best_epoch, cheap_best_val_original, cheap_best_val_norm = train_residual_regressor(
        cheap,
        train,
        val,
        train_x,
        train_y,
        val_x,
        val_y,
        stats,
        epochs=args.cheap_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        patience=args.patience,
        name="cheap_residual",
    )

    expensive_history, expensive_best_epoch, expensive_best_val_original, expensive_best_val_norm = train_residual_regressor(
        expensive,
        train,
        val,
        train_x,
        train_y,
        val_x,
        val_y,
        stats,
        epochs=args.expensive_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        patience=args.patience,
        name="expensive_residual",
    )

    split_data = {"train": train, "val": val}
    split_x = {"train": train_x, "val": val_x}
    if test is not None:
        split_data["test"] = test
        split_x["test"] = test_x

    all_rows = []
    summary = {}

    for split_name, data in split_data.items():
        base = baseline_predictions(data, stats)

        cheap_err, cheap_pred = evaluate_model(cheap, data, split_x[split_name], stats, args.batch_size, args.device)
        expensive_err, expensive_pred = evaluate_model(expensive, data, split_x[split_name], stats, args.batch_size, args.device)

        preds = {
            **base,
            "cheap_residual": cheap_pred,
            "expensive_residual": expensive_pred,
        }

        rows = group_metrics(split_name, data, preds)
        all_rows.extend(rows)

        summary[split_name] = {
            "identity_error": mean_mse(base["identity"], data["z_future"]),
            "train_mean_delta_error": mean_mse(base["train_mean_delta"], data["z_future"]),
            "cheap_residual_error": cheap_err,
            "expensive_residual_error": expensive_err,
            "cheap_improvement_vs_identity": mean_mse(base["identity"], data["z_future"]) - cheap_err,
            "expensive_improvement_vs_identity": mean_mse(base["identity"], data["z_future"]) - expensive_err,
            "gap_cheap_minus_expensive": cheap_err - expensive_err,
        }

    write_csv(out / "residual_predictor_diagnostics.csv", all_rows)
    write_markdown(out / "residual_predictor_diagnostics.md", all_rows)

    metrics = {
        "seed": args.seed,
        "train": args.train,
        "val": args.val,
        "test": args.test,
        "center_target": args.center_target,
        "model": {
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "cheap_hidden": args.cheap_hidden,
            "cheap_layers": args.cheap_layers,
            "expensive_hidden": args.expensive_hidden,
            "expensive_layers": args.expensive_layers,
            "dropout": args.dropout,
            "zero_init_final": True,
            "prediction_type": "residual_delta",
            "target": "z_future_minus_z_current",
        },
        "training": {
            "cheap_best_epoch": cheap_best_epoch,
            "cheap_best_val_original": cheap_best_val_original,
            "cheap_best_val_norm": cheap_best_val_norm,
            "expensive_best_epoch": expensive_best_epoch,
            "expensive_best_val_original": expensive_best_val_original,
            "expensive_best_val_norm": expensive_best_val_norm,
            "cheap_history": cheap_history,
            "expensive_history": expensive_history,
        },
        "summary": summary,
        "stats": {
            "x_mean_shape": list(stats["x_mean"].shape),
            "x_std_min": float(stats["x_std"].min()),
            "x_std_max": float(stats["x_std"].max()),
            "delta_std_min": float(stats["delta_std"].min()),
            "delta_std_max": float(stats["delta_std"].max()),
            "train_mean_delta_norm": float(np.linalg.norm(stats["train_mean_delta"])),
        },
    }

    with (out / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    torch.save(
        {
            "cheap": cheap.state_dict(),
            "expensive": expensive.state_dict(),
            "metrics": metrics,
            "stats": {k: torch.from_numpy(v) for k, v in stats.items()},
        },
        out / "checkpoint.pt",
    )

    print(out / "metrics.json")
    print(out / "residual_predictor_diagnostics.md")
    print("\nSummary:")
    for split, vals in summary.items():
        print(f"\n[{split}]")
        for k, v in vals.items():
            print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class CheapPredictor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--val-input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--val-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, default=Path("outputs/error_reliability_label_report.json"))
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hard-fraction", type=float, default=0.30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32)).to(device)


def compute_errors(
    model: nn.Module,
    data: dict[str, np.ndarray],
    device: str,
    batch_size: int,
) -> np.ndarray:
    model.eval()

    z = tensor(data["z_current"], device)
    a = tensor(data["action"], device)
    y = tensor(data["z_future"], device)

    loader = DataLoader(TensorDataset(z, a, y), batch_size=batch_size, shuffle=False)
    errors = []

    with torch.no_grad():
        for zb, ab, yb in loader:
            pred = model(zb, ab)
            err = ((pred - yb) ** 2).mean(dim=-1)
            errors.append(err.detach().cpu().numpy())

    return np.concatenate(errors, axis=0).astype(np.float32)


def write_relabelled(
    source: dict[str, np.ndarray],
    output: Path,
    errors: np.ndarray,
    threshold: float,
) -> None:
    out = {}

    for k, v in source.items():
        out[k] = v

    # Replace the previous heuristic target.
    out["expected_unreliable"] = (errors >= threshold).astype(np.float32)

    # Keep useful diagnostics inside the NPZ.
    out["cheap_prediction_error"] = errors.astype(np.float32)
    out["error_reliability_threshold"] = np.asarray(threshold, dtype=np.float32)
    out["reliability_target_kind"] = np.asarray("cheap_predictor_error_quantile")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **out)

    print(f"Wrote {output}")
    print(f"  samples={len(errors)} positive_rate={out['expected_unreliable'].mean():.3f}")


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train = load_npz(args.train_input)
    val = load_npz(args.val_input)

    latent_dim = int(train["z_current"].shape[1])
    action_dim = int(train["action"].shape[1])

    model = CheapPredictor(latent_dim, action_dim, args.hidden_dim).to(args.device)

    z_train = tensor(train["z_current"], args.device)
    a_train = tensor(train["action"], args.device)
    y_train = tensor(train["z_future"], args.device)

    loader = DataLoader(
        TensorDataset(z_train, a_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for zb, ab, yb in loader:
            pred = model(zb, ab)
            loss = ((pred - yb) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "cheap_prediction_loss": mean_loss})
        print(f"epoch={epoch:03d} cheap_prediction_loss={mean_loss:.6f}")

    train_errors = compute_errors(model, train, args.device, args.batch_size)
    val_errors = compute_errors(model, val, args.device, args.batch_size)

    threshold = float(np.quantile(train_errors, 1.0 - args.hard_fraction))

    print(f"threshold={threshold:.6f}")
    print(f"train error mean={train_errors.mean():.6f} std={train_errors.std():.6f}")
    print(f"val   error mean={val_errors.mean():.6f} std={val_errors.std():.6f}")

    write_relabelled(train, args.train_output, train_errors, threshold)
    write_relabelled(val, args.val_output, val_errors, threshold)

    report = {
        "latent_dim": latent_dim,
        "action_dim": action_dim,
        "hard_fraction": args.hard_fraction,
        "threshold": threshold,
        "train_error_mean": float(train_errors.mean()),
        "train_error_std": float(train_errors.std()),
        "val_error_mean": float(val_errors.mean()),
        "val_error_std": float(val_errors.std()),
        "train_positive_rate": float((train_errors >= threshold).mean()),
        "val_positive_rate": float((val_errors >= threshold).mean()),
        "history": history,
    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    with args.report_output.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote report to {args.report_output}")


if __name__ == "__main__":
    main()

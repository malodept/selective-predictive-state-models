from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLPPredictor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        blocks = []
        in_dim = latent_dim + action_dim

        for i in range(layers):
            blocks.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            blocks.append(nn.GELU())

        blocks.append(nn.Linear(hidden_dim, latent_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))


class ReliabilityHead(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        in_dim = latent_dim + action_dim + latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor, zhat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a, zhat], dim=-1)).squeeze(-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--cheap-hidden", type=int, default=128)
    p.add_argument("--cheap-layers", type=int, default=2)

    p.add_argument("--expensive-hidden", type=int, default=512)
    p.add_argument("--expensive-layers", type=int, default=5)

    p.add_argument("--reliability-hidden", type=int, default=256)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--cheap-epochs", type=int, default=None)
    p.add_argument("--expensive-epochs", type=int, default=None)
    p.add_argument("--reliability-epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--hard-fraction", type=float, default=0.30)
    p.add_argument("--lambda-compute", type=float, default=0.04)
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90])

    p.add_argument("--cheap-compute", type=float, default=1.0)
    p.add_argument("--expensive-compute", type=float, default=4.0)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def as_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32)).to(device)


def make_loader(data: dict[str, np.ndarray], device: str, batch_size: int, shuffle: bool) -> DataLoader:
    z = as_tensor(data["z_current"], device)
    a = as_tensor(data["action"], device)
    y = as_tensor(data["z_future"], device)
    return DataLoader(TensorDataset(z, a, y), batch_size=batch_size, shuffle=shuffle)


def train_predictor(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> list[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for z, a, y in train_loader:
            pred = model(z, a)
            loss = ((pred - y) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(losses))
        history.append(mean_loss)
        print(f"predictor epoch={epoch:03d} loss={mean_loss:.6f}")

    return history


@torch.no_grad()
def predict_and_errors(model: nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    errors = []

    for z, a, y in loader:
        pred = model(z, a)
        err = ((pred - y) ** 2).mean(dim=-1)
        preds.append(pred.detach().cpu().numpy())
        errors.append(err.detach().cpu().numpy())

    return np.concatenate(preds, axis=0), np.concatenate(errors, axis=0)


def train_reliability(
    model: ReliabilityHead,
    cheap_model: nn.Module,
    train_data: dict[str, np.ndarray],
    threshold: float,
    args: argparse.Namespace,
) -> list[float]:
    z = as_tensor(train_data["z_current"], args.device)
    a = as_tensor(train_data["action"], args.device)
    y = as_tensor(train_data["z_future"], args.device)

    with torch.no_grad():
        zhat = cheap_model(z, a)
        err = ((zhat - y) ** 2).mean(dim=-1)
        target = (err >= threshold).float()

    loader = DataLoader(TensorDataset(z, a, zhat.detach(), target), batch_size=args.batch_size, shuffle=True)

    pos_rate = float(target.mean().detach().cpu())
    pos_weight = torch.tensor([(1.0 - pos_rate) / max(pos_rate, 1e-6)], device=args.device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    reliability_epochs = args.reliability_epochs or args.epochs

    for epoch in range(1, reliability_epochs + 1):
        model.train()
        losses = []

        for zb, ab, phatb, tb in loader:
            logits = model(zb, ab, phatb)
            loss = loss_fn(logits, tb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))

        mean_loss = float(np.mean(losses))
        history.append(mean_loss)
        print(f"reliability epoch={epoch:03d} loss={mean_loss:.6f}")

    return history


@torch.no_grad()
def evaluate(
    cheap: nn.Module,
    expensive: nn.Module,
    reliability: ReliabilityHead,
    val_loader: DataLoader,
    args: argparse.Namespace,
) -> dict:
    cheap.eval()
    expensive.eval()
    reliability.eval()

    cheap_errors = []
    expensive_errors = []
    scores = []

    for z, a, y in val_loader:
        cheap_pred = cheap(z, a)
        expensive_pred = expensive(z, a)

        cheap_err = ((cheap_pred - y) ** 2).mean(dim=-1)
        expensive_err = ((expensive_pred - y) ** 2).mean(dim=-1)

        score = torch.sigmoid(reliability(z, a, cheap_pred))

        cheap_errors.append(cheap_err.cpu().numpy())
        expensive_errors.append(expensive_err.cpu().numpy())
        scores.append(score.cpu().numpy())

    cheap_errors = np.concatenate(cheap_errors)
    expensive_errors = np.concatenate(expensive_errors)
    scores = np.concatenate(scores)

    rows = []

    cheap_row = {
        "policy": "cheap-only",
        "threshold": None,
        "mean_error": float(cheap_errors.mean()),
        "mean_compute": args.cheap_compute,
        "selected_fraction": 0.0,
    }
    cheap_row["utility"] = -cheap_row["mean_error"] - args.lambda_compute * cheap_row["mean_compute"]
    rows.append(cheap_row)

    expensive_row = {
        "policy": "all-expensive",
        "threshold": None,
        "mean_error": float(expensive_errors.mean()),
        "mean_compute": args.expensive_compute,
        "selected_fraction": 1.0,
    }
    expensive_row["utility"] = -expensive_row["mean_error"] - args.lambda_compute * expensive_row["mean_compute"]
    rows.append(expensive_row)

    for tau in args.thresholds:
        select = scores >= tau
        selected_fraction = float(select.mean())

        mixed_errors = np.where(select, expensive_errors, cheap_errors)
        mean_compute = args.cheap_compute + selected_fraction * (args.expensive_compute - args.cheap_compute)
        mean_error = float(mixed_errors.mean())
        utility = -mean_error - args.lambda_compute * mean_compute

        rows.append({
            "policy": f"threshold={tau:.2f}",
            "threshold": float(tau),
            "mean_error": mean_error,
            "mean_compute": float(mean_compute),
            "selected_fraction": selected_fraction,
            "utility": float(utility),
        })

    best = max(rows, key=lambda r: r["utility"])

    return {
        "cheap_error_mean": float(cheap_errors.mean()),
        "expensive_error_mean": float(expensive_errors.mean()),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "lambda_compute": args.lambda_compute,
        "rows": rows,
        "best": best,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train = load_npz(args.train)
    val = load_npz(args.val)

    latent_dim = int(train["z_current"].shape[1])
    action_dim = int(train["action"].shape[1])

    train_loader = make_loader(train, args.device, args.batch_size, shuffle=True)
    val_loader = make_loader(val, args.device, args.batch_size, shuffle=False)

    print(f"latent_dim={latent_dim} action_dim={action_dim}")

    cheap = MLPPredictor(latent_dim, action_dim, args.cheap_hidden, args.cheap_layers).to(args.device)
    expensive = MLPPredictor(latent_dim, action_dim, args.expensive_hidden, args.expensive_layers).to(args.device)

    print("Training cheap predictor")
    cheap_epochs = args.cheap_epochs or args.epochs
    expensive_epochs = args.expensive_epochs or args.epochs
    reliability_epochs = args.reliability_epochs or args.epochs

    cheap_history = train_predictor(cheap, train_loader, cheap_epochs, args.lr, args.weight_decay)

    print("Training expensive predictor")
    expensive_history = train_predictor(expensive, train_loader, expensive_epochs, args.lr, args.weight_decay)

    _, train_cheap_errors = predict_and_errors(cheap, train_loader)
    threshold = float(np.quantile(train_cheap_errors, 1.0 - args.hard_fraction))
    print(f"cheap error threshold={threshold:.6f}")

    reliability = ReliabilityHead(latent_dim, action_dim, args.reliability_hidden).to(args.device)

    print("Training reliability head")
    reliability_history = train_reliability(reliability, cheap, train, threshold, args)

    eval_result = evaluate(cheap, expensive, reliability, val_loader, args)

    report = {
        "latent_dim": latent_dim,
        "action_dim": action_dim,
        "cheap_hidden": args.cheap_hidden,
        "cheap_layers": args.cheap_layers,
        "expensive_hidden": args.expensive_hidden,
        "expensive_layers": args.expensive_layers,
        "hard_fraction": args.hard_fraction,
        "cheap_error_threshold": threshold,
        "cheap_history": cheap_history,
        "expensive_history": expensive_history,
        "reliability_history": reliability_history,
        "eval": eval_result,
    }

    out = args.output_dir / "real_refinement_metrics.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(eval_result, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

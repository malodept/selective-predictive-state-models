from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_envsplit_residual_predictor import MLP, load_npz, feature_matrix, mean_mse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-alphas", type=int, default=101)
    return p.parse_args()


def to_numpy_stats(stats: dict) -> dict[str, np.ndarray]:
    out = {}
    for k, v in stats.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().numpy().astype(np.float32)
        else:
            out[k] = np.asarray(v, dtype=np.float32)
    return out


def transform_x(data: dict[str, np.ndarray], stats: dict[str, np.ndarray]) -> np.ndarray:
    x = feature_matrix(data)
    return ((x - stats["x_mean"]) / stats["x_std"]).astype(np.float32)


@torch.no_grad()
def predict_delta_norm(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    outs = []
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)

    for xb in loader:
        xb = xb.to(device)
        out = model(xb).detach().cpu().numpy().astype(np.float32)
        outs.append(out)

    return np.concatenate(outs, axis=0)


def load_model(ckpt: dict, name: str, metrics: dict, device: str) -> MLP:
    cfg = metrics["model"]

    if name == "cheap":
        hidden = int(cfg["cheap_hidden"])
        layers = int(cfg["cheap_layers"])
    elif name == "expensive":
        hidden = int(cfg["expensive_hidden"])
        layers = int(cfg["expensive_layers"])
    else:
        raise ValueError(name)

    model = MLP(
        input_dim=int(cfg["input_dim"]),
        output_dim=int(cfg["latent_dim"]),
        hidden_dim=hidden,
        layers=layers,
        dropout=float(cfg.get("dropout", 0.0)),
        zero_init_final=True,
    ).to(device)

    model.load_state_dict(ckpt[name])
    model.eval()
    return model


def delta_hat_from_model(
    model: torch.nn.Module,
    data: dict[str, np.ndarray],
    stats: dict[str, np.ndarray],
    batch_size: int,
    device: str,
) -> np.ndarray:
    x = transform_x(data, stats)
    pred_norm = predict_delta_norm(model, x, batch_size, device)
    return pred_norm * stats["delta_std"][None, :] + stats["delta_mean"][None, :]


def errors_for_alphas(
    data: dict[str, np.ndarray],
    delta_hat: np.ndarray,
    alphas: np.ndarray,
) -> np.ndarray:
    z0 = data["z_current"].astype(np.float32)
    y = data["z_future"].astype(np.float32)

    errors = []
    for a in alphas:
        pred = z0 + float(a) * delta_hat
        errors.append(mean_mse(pred, y))

    return np.asarray(errors, dtype=np.float64)


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_md(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Residual shrinkage alpha diagnostic",
        "",
        "Prediction family: `z_pred(alpha) = z_current + alpha * delta_hat`, with `alpha in [0, 1]`.",
        "",
        "| split | model | rule | alpha | error | identity error | improvement vs identity | ratio vs identity |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['split']} | {r['model']} | {r['rule']} | "
            f"{r['alpha']:.4f} | {r['error']:.6f} | {r['identity_error']:.6f} | "
            f"{r['improvement_vs_identity']:.6f} | {r['ratio_vs_identity']:.6f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    metrics = ckpt["metrics"]
    stats = to_numpy_stats(ckpt["stats"])

    val = load_npz(args.val)
    test = load_npz(args.test)

    cheap = load_model(ckpt, "cheap", metrics, args.device)
    expensive = load_model(ckpt, "expensive", metrics, args.device)

    alphas = np.linspace(0.0, 1.0, args.num_alphas)

    delta = {
        "cheap_residual": {
            "val": delta_hat_from_model(cheap, val, stats, args.batch_size, args.device),
            "test": delta_hat_from_model(cheap, test, stats, args.batch_size, args.device),
        },
        "expensive_residual": {
            "val": delta_hat_from_model(expensive, val, stats, args.batch_size, args.device),
            "test": delta_hat_from_model(expensive, test, stats, args.batch_size, args.device),
        },
    }

    data_by_split = {"val": val, "test": test}
    rows = []
    selected_alpha = {}

    for model_name in ["cheap_residual", "expensive_residual"]:
        val_errors = errors_for_alphas(val, delta[model_name]["val"], alphas)
        best_idx = int(np.argmin(val_errors))
        selected_alpha[model_name] = float(alphas[best_idx])

    for split_name, data in data_by_split.items():
        identity_error = mean_mse(data["z_current"], data["z_future"])

        for model_name in ["cheap_residual", "expensive_residual"]:
            errors = errors_for_alphas(data, delta[model_name][split_name], alphas)

            rules = []

            # alpha=1 is the raw residual predictor already reported before.
            raw_idx = int(np.argmin(np.abs(alphas - 1.0)))
            rules.append(("raw_alpha_1", float(alphas[raw_idx]), float(errors[raw_idx])))

            # oracle per split is diagnostic only.
            oracle_idx = int(np.argmin(errors))
            rules.append(("oracle_best_alpha_on_this_split", float(alphas[oracle_idx]), float(errors[oracle_idx])))

            # val-selected alpha is the deployable rule for test.
            a_val = selected_alpha[model_name]
            val_idx = int(np.argmin(np.abs(alphas - a_val)))
            rules.append(("alpha_selected_on_val", float(alphas[val_idx]), float(errors[val_idx])))

            for rule, alpha, err in rules:
                rows.append(
                    {
                        "split": split_name,
                        "model": model_name,
                        "rule": rule,
                        "alpha": alpha,
                        "error": err,
                        "identity_error": identity_error,
                        "improvement_vs_identity": identity_error - err,
                        "ratio_vs_identity": err / identity_error if identity_error > 0 else float("nan"),
                    }
                )

    write_csv(args.output_dir / "residual_shrinkage_alpha.csv", rows)
    write_md(args.output_dir / "residual_shrinkage_alpha.md", rows)

    print(args.output_dir / "residual_shrinkage_alpha.md")
    print((args.output_dir / "residual_shrinkage_alpha.md").read_text())

    print("\nSelected alphas from val:")
    for k, v in selected_alpha.items():
        print(k, v)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_envsplit_residual_predictor import MLP, load_npz, feature_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    p.add_argument("--shrinkage-csvs", nargs="+", type=Path, required=True)
    p.add_argument("--model", choices=["cheap_residual", "expensive_residual"], default="cheap_residual")
    p.add_argument("--out-dir", type=Path, default=Path("reports/tables/protocol/residual_shrinkage_trajectory_bootstrap"))
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bootstrap-iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
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


def load_model(ckpt: dict, model_name: str, metrics: dict, device: str) -> MLP:
    cfg = metrics["model"]

    if model_name == "cheap_residual":
        state_key = "cheap"
        hidden = int(cfg["cheap_hidden"])
        layers = int(cfg["cheap_layers"])
    elif model_name == "expensive_residual":
        state_key = "expensive"
        hidden = int(cfg["expensive_hidden"])
        layers = int(cfg["expensive_layers"])
    else:
        raise ValueError(model_name)

    model = MLP(
        input_dim=int(cfg["input_dim"]),
        output_dim=int(cfg["latent_dim"]),
        hidden_dim=hidden,
        layers=layers,
        dropout=float(cfg.get("dropout", 0.0)),
        zero_init_final=True,
    ).to(device)

    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model


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


def per_sample_mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean((a - b) ** 2, axis=1)


def read_alpha(path: Path, model_name: str) -> float:
    df = pd.read_csv(path)
    row = df[
        (df["split"] == "test")
        & (df["model"] == model_name)
        & (df["rule"] == "alpha_selected_on_val")
    ]
    if len(row) != 1:
        raise ValueError(f"Could not find alpha row in {path} for {model_name}")
    return float(row.iloc[0]["alpha"])


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, iters: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    means = np.empty(iters, dtype=np.float64)

    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()

    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    if len(args.checkpoints) != len(args.shrinkage_csvs):
        raise ValueError("Need same number of checkpoints and shrinkage CSVs.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    test = load_npz(args.test)
    z0 = test["z_current"].astype(np.float32)
    y = test["z_future"].astype(np.float32)
    trajectory = test["trajectory_id"].astype(str)

    identity_err = per_sample_mse(z0, y)

    detailed_rows = []

    for seed_idx, (ckpt_path, csv_path) in enumerate(zip(args.checkpoints, args.shrinkage_csvs)):
        ckpt = torch.load(ckpt_path, map_location=args.device)
        metrics = ckpt["metrics"]
        stats = to_numpy_stats(ckpt["stats"])
        alpha = read_alpha(csv_path, args.model)

        model = load_model(ckpt, args.model, metrics, args.device)
        x = transform_x(test, stats)

        pred_norm = predict_delta_norm(model, x, args.batch_size, args.device)
        delta_hat = pred_norm * stats["delta_std"][None, :] + stats["delta_mean"][None, :]
        pred = z0 + alpha * delta_hat

        model_err = per_sample_mse(pred, y)
        improvement = identity_err - model_err

        for tid in sorted(set(trajectory)):
            mask = trajectory == tid
            detailed_rows.append(
                {
                    "seed": seed_idx,
                    "model": args.model,
                    "alpha": alpha,
                    "trajectory_id": tid,
                    "samples": int(mask.sum()),
                    "identity_error": float(identity_err[mask].mean()),
                    "model_error": float(model_err[mask].mean()),
                    "improvement_vs_identity": float(improvement[mask].mean()),
                    "ratio_vs_identity": float(model_err[mask].mean() / identity_err[mask].mean()),
                }
            )

    detailed = pd.DataFrame(detailed_rows)
    detailed_path = args.out_dir / f"{args.model}_trajectory_effects.csv"
    detailed.to_csv(detailed_path, index=False)

    # Equal-weight each seed-trajectory pair.
    values = detailed["improvement_vs_identity"].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    mean, lo, hi = bootstrap_ci(values, rng, args.bootstrap_iters)

    # Equal-weight trajectories after averaging over seeds.
    by_traj = detailed.groupby("trajectory_id", as_index=False).agg(
        improvement_vs_identity=("improvement_vs_identity", "mean"),
        identity_error=("identity_error", "mean"),
        model_error=("model_error", "mean"),
        ratio_vs_identity=("ratio_vs_identity", "mean"),
        samples=("samples", "mean"),
    )

    traj_values = by_traj["improvement_vs_identity"].to_numpy(dtype=np.float64)
    traj_mean, traj_lo, traj_hi = bootstrap_ci(traj_values, rng, args.bootstrap_iters)

    summary_path = args.out_dir / f"{args.model}_trajectory_bootstrap.md"

    lines = [
        f"# Trajectory-level bootstrap for {args.model}",
        "",
        "Metric: per-trajectory mean of `identity_error - calibrated_residual_error` on the held-out test environment.",
        "",
        f"Seed-trajectory units: `{len(values)}`.",
        f"Trajectory units after seed averaging: `{len(traj_values)}`.",
        "",
        "| aggregation | mean improvement | 95% bootstrap CI low | 95% bootstrap CI high |",
        "| --- | ---: | ---: | ---: |",
        f"| seed-trajectory equal weight | {mean:.6f} | {lo:.6f} | {hi:.6f} |",
        f"| trajectory equal weight | {traj_mean:.6f} | {traj_lo:.6f} | {traj_hi:.6f} |",
        "",
        "## Per-trajectory effects averaged over seeds",
        "",
        "| trajectory | samples | identity error | model error | improvement vs identity | ratio vs identity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, r in by_traj.sort_values("improvement_vs_identity", ascending=False).iterrows():
        lines.append(
            f"| {r['trajectory_id']} | {int(r['samples'])} | "
            f"{r['identity_error']:.6f} | {r['model_error']:.6f} | "
            f"{r['improvement_vs_identity']:.6f} | {r['ratio_vs_identity']:.6f} |"
        )

    summary_path.write_text("\n".join(lines) + "\n")

    print(summary_path)
    print(summary_path.read_text())


if __name__ == "__main__":
    main()

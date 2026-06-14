from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_conditional_residual_shrinkage import (
    load_npz,
    torch_stats_to_numpy,
    normalized_input,
    build_model,
    predict_delta,
    oracle_alpha,
    mean_mse,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--checkpoint-template", type=str, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--model", choices=["cheap_residual", "expensive_residual"], default="cheap_residual")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def mse_pred(z0: np.ndarray, y: np.ndarray, delta: np.ndarray, alpha: float | np.ndarray) -> float:
    if np.isscalar(alpha):
        pred = z0 + float(alpha) * delta
    else:
        pred = z0 + alpha[:, None].astype(np.float32) * delta
    return mean_mse(pred, y)


def best_alpha_closed_form(z0: np.ndarray, y: np.ndarray, delta: np.ndarray, mask: np.ndarray | None = None) -> float:
    true_delta = y - z0
    if mask is not None:
        true_delta = true_delta[mask]
        delta = delta[mask]
    num = float(np.sum(delta * true_delta))
    den = float(np.sum(delta * delta)) + 1e-12
    return float(np.clip(num / den, 0.0, 1.0))


def oracle_alpha_for_delta(z0: np.ndarray, y: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return oracle_alpha(delta, y - z0)


def make_scores(delta_stack: np.ndarray, delta_mean: np.ndarray) -> dict[str, np.ndarray]:
    # delta_stack: seeds x samples x dim
    var = np.var(delta_stack, axis=0)
    disagreement = np.sqrt(np.mean(var, axis=1))
    delta_norm = np.linalg.norm(delta_mean, axis=1)
    relative_disagreement = disagreement / (delta_norm + 1e-8)

    # Average pairwise distance between seed residuals.
    if delta_stack.shape[0] >= 2:
        pair_dists = []
        for i in range(delta_stack.shape[0]):
            for j in range(i + 1, delta_stack.shape[0]):
                pair_dists.append(np.linalg.norm(delta_stack[i] - delta_stack[j], axis=1))
        pairwise_disagreement = np.mean(np.stack(pair_dists, axis=0), axis=0)
    else:
        pairwise_disagreement = disagreement

    return {
        "disagreement": disagreement.astype(np.float32),
        "relative_disagreement": relative_disagreement.astype(np.float32),
        "pairwise_disagreement": pairwise_disagreement.astype(np.float32),
        "delta_norm": delta_norm.astype(np.float32),
    }


def fit_score_bin_calibrator(
    z0_val: np.ndarray,
    y_val: np.ndarray,
    delta_val: np.ndarray,
    score_val: np.ndarray,
    alpha_global: float,
) -> dict:
    best = {
        "score_name": "",
        "n_bins": 0,
        "gamma": 0.0,
        "edges": None,
        "bin_alphas": None,
        "val_error": mse_pred(z0_val, y_val, delta_val, alpha_global),
        "alpha_mean": alpha_global,
        "alpha_std": 0.0,
    }

    for n_bins in [2, 3, 4, 5, 8, 10]:
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(score_val, qs)
        edges[0] = -np.inf
        edges[-1] = np.inf
        bins = np.digitize(score_val, edges[1:-1], right=True)

        bin_alphas = np.zeros(n_bins, dtype=np.float32)
        for b in range(n_bins):
            mask = bins == b
            if int(mask.sum()) < 10:
                bin_alphas[b] = alpha_global
            else:
                bin_alphas[b] = best_alpha_closed_form(z0_val, y_val, delta_val, mask)

        alpha_bin = bin_alphas[bins]

        for gamma in np.linspace(0.0, 1.0, 101):
            alpha = np.clip((1.0 - gamma) * alpha_global + gamma * alpha_bin, 0.0, 1.0)
            err = mse_pred(z0_val, y_val, delta_val, alpha)
            if err < best["val_error"]:
                best = {
                    "score_name": "",
                    "n_bins": int(n_bins),
                    "gamma": float(gamma),
                    "edges": edges,
                    "bin_alphas": bin_alphas,
                    "val_error": float(err),
                    "alpha_mean": float(alpha.mean()),
                    "alpha_std": float(alpha.std()),
                }

    return best


def apply_calibrator(score: np.ndarray, alpha_global: float, calib: dict) -> np.ndarray:
    if calib["n_bins"] == 0:
        return np.full_like(score, alpha_global, dtype=np.float32)

    bins = np.digitize(score, calib["edges"][1:-1], right=True)
    alpha_bin = calib["bin_alphas"][bins]
    alpha = np.clip((1.0 - calib["gamma"]) * alpha_global + calib["gamma"] * alpha_bin, 0.0, 1.0)
    return alpha.astype(np.float32)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    x0 = xr - xr.mean()
    y0 = yr - yr.mean()
    return float(np.sum(x0 * y0) / (np.sqrt(np.sum(x0*x0) * np.sum(y0*y0)) + 1e-12))


def row(method, split, z0, y, delta, alpha, alpha_oracle, extra=None):
    identity_error = mean_mse(z0, y)
    err = mse_pred(z0, y, delta, alpha)

    if np.isscalar(alpha):
        alpha_mean = float(alpha)
        alpha_std = 0.0
        alpha_spearman = np.nan
    else:
        alpha_mean = float(np.mean(alpha))
        alpha_std = float(np.std(alpha))
        alpha_spearman = spearman(alpha, alpha_oracle)

    out = {
        "split": split,
        "method": method,
        "error": err,
        "identity_error": identity_error,
        "improvement_vs_identity": identity_error - err,
        "ratio_vs_identity": err / identity_error,
        "alpha_mean": alpha_mean,
        "alpha_std": alpha_std,
        "alpha_spearman_oracle": alpha_spearman,
    }
    if extra:
        out.update(extra)
    return out


def pm(s):
    if pd.Series(s).isna().all():
        return "nan"
    return f"{pd.Series(s).mean():.6f} ± {pd.Series(s).std(ddof=1):.6f}" if len(s) > 1 else f"{pd.Series(s).mean():.6f}"


def write_md(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "# Ensemble uncertainty shrinkage diagnostic",
        "",
        "This diagnostic uses the disagreement of independently trained residual predictors as a deployment-time uncertainty signal for residual shrinkage.",
        "",
        "| split | method | error | identity error | improvement vs identity | ratio vs identity | alpha mean | alpha std | alpha Spearman oracle | score | n bins | gamma |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]

    for _, r in df.iterrows():
        lines.append(
            f"| {r['split']} | {r['method']} | "
            f"{r['error']:.6f} | {r['identity_error']:.6f} | {r['improvement_vs_identity']:.6f} | "
            f"{r['ratio_vs_identity']:.6f} | {r['alpha_mean']:.6f} | {r['alpha_std']:.6f} | "
            f"{r['alpha_spearman_oracle'] if not pd.isna(r['alpha_spearman_oracle']) else 'nan'} | "
            f"{r.get('score_name', '')} | {r.get('n_bins', 0)} | {r.get('gamma', 0.0):.4f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    val = load_npz(args.val)
    test = load_npz(args.test)

    deltas_val = []
    deltas_test = []

    for seed in args.seeds:
        print(f"Loading seed {seed}")
        ckpt = torch.load(Path(args.checkpoint_template.format(seed=seed)), map_location=args.device)
        stats = torch_stats_to_numpy(ckpt["stats"])
        model = build_model(ckpt, args.model, args.device)

        deltas_val.append(
            predict_delta(model, normalized_input(val, stats), stats, args.batch_size, args.device)
        )
        deltas_test.append(
            predict_delta(model, normalized_input(test, stats), stats, args.batch_size, args.device)
        )

    stack_val = np.stack(deltas_val, axis=0)
    stack_test = np.stack(deltas_test, axis=0)

    delta_val_mean = stack_val.mean(axis=0).astype(np.float32)
    delta_test_mean = stack_test.mean(axis=0).astype(np.float32)

    z0_val = val["z_current"].astype(np.float32)
    y_val = val["z_future"].astype(np.float32)
    z0_test = test["z_current"].astype(np.float32)
    y_test = test["z_future"].astype(np.float32)

    alpha_global = best_alpha_closed_form(z0_val, y_val, delta_val_mean)

    alpha_oracle_val = oracle_alpha_for_delta(z0_val, y_val, delta_val_mean)
    alpha_oracle_test = oracle_alpha_for_delta(z0_test, y_test, delta_test_mean)

    scores_val = make_scores(stack_val, delta_val_mean)
    scores_test = make_scores(stack_test, delta_test_mean)

    best_score_name = None
    best_calib = None

    for name, score_val in scores_val.items():
        calib = fit_score_bin_calibrator(z0_val, y_val, delta_val_mean, score_val, alpha_global)
        calib["score_name"] = name

        if best_calib is None or calib["val_error"] < best_calib["val_error"]:
            best_calib = calib
            best_score_name = name

    alpha_val_unc = apply_calibrator(scores_val[best_score_name], alpha_global, best_calib)
    alpha_test_unc = apply_calibrator(scores_test[best_score_name], alpha_global, best_calib)

    rows = []
    rows.append(row("identity", "val", z0_val, y_val, delta_val_mean, 0.0, alpha_oracle_val))
    rows.append(row("ensemble_raw_alpha_1", "val", z0_val, y_val, delta_val_mean, 1.0, alpha_oracle_val))
    rows.append(row("ensemble_global_alpha_val", "val", z0_val, y_val, delta_val_mean, alpha_global, alpha_oracle_val))
    rows.append(row("ensemble_uncertainty_calibrated", "val", z0_val, y_val, delta_val_mean, alpha_val_unc, alpha_oracle_val, best_calib))
    rows.append(row("ensemble_oracle_per_sample_alpha", "val", z0_val, y_val, delta_val_mean, alpha_oracle_val, alpha_oracle_val))

    rows.append(row("identity", "test", z0_test, y_test, delta_test_mean, 0.0, alpha_oracle_test))
    rows.append(row("ensemble_raw_alpha_1", "test", z0_test, y_test, delta_test_mean, 1.0, alpha_oracle_test))
    rows.append(row("ensemble_global_alpha_val", "test", z0_test, y_test, delta_test_mean, alpha_global, alpha_oracle_test))
    rows.append(row("ensemble_uncertainty_calibrated", "test", z0_test, y_test, delta_test_mean, alpha_test_unc, alpha_oracle_test, best_calib))
    rows.append(row("ensemble_oracle_per_sample_alpha", "test", z0_test, y_test, delta_test_mean, alpha_oracle_test, alpha_oracle_test))

    df = pd.DataFrame(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "ensemble_uncertainty_shrinkage.csv"
    md_path = args.out_dir / "ensemble_uncertainty_shrinkage.md"

    df.to_csv(csv_path, index=False)
    write_md(md_path, df)

    print("Selected uncertainty score:", best_score_name)
    print("Global alpha:", alpha_global)
    print(md_path)
    print(md_path.read_text())


if __name__ == "__main__":
    main()

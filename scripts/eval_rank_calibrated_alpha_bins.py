from __future__ import annotations

import argparse
import re
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
    compact_alpha_features,
    oracle_alpha,
    mean_mse,
    select_global_alpha,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--checkpoint-template", type=str, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--model", choices=["cheap_residual", "expensive_residual"], default="cheap_residual")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def mse_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((pred - target) ** 2, axis=1)


def error_for_alpha(z0: np.ndarray, y: np.ndarray, delta_hat: np.ndarray, alpha: float | np.ndarray) -> float:
    if np.isscalar(alpha):
        pred = z0 + float(alpha) * delta_hat
    else:
        pred = z0 + alpha[:, None].astype(np.float32) * delta_hat
    return mean_mse(pred, y)


def best_alpha_closed_form(z0: np.ndarray, y: np.ndarray, delta_hat: np.ndarray, mask: np.ndarray | None = None) -> float:
    true_delta = y - z0
    if mask is not None:
        true_delta = true_delta[mask]
        delta_hat = delta_hat[mask]

    num = float(np.sum(delta_hat * true_delta))
    den = float(np.sum(delta_hat * delta_hat)) + 1e-12
    return float(np.clip(num / den, 0.0, 1.0))


def make_quantile_bins(alpha_val: np.ndarray, n_bins: int) -> np.ndarray:
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(alpha_val, qs)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_bins(alpha: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.digitize(alpha, edges[1:-1], right=True)


def fit_bin_alphas(
    z0_val: np.ndarray,
    y_val: np.ndarray,
    delta_val: np.ndarray,
    alpha_val_raw: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = make_quantile_bins(alpha_val_raw, n_bins)
    bins = apply_bins(alpha_val_raw, edges)

    bin_alphas = np.zeros(n_bins, dtype=np.float32)

    global_alpha = best_alpha_closed_form(z0_val, y_val, delta_val)

    for b in range(n_bins):
        mask = bins == b
        if int(mask.sum()) < 10:
            bin_alphas[b] = global_alpha
        else:
            bin_alphas[b] = best_alpha_closed_form(z0_val, y_val, delta_val, mask)

    return edges, bin_alphas


def select_rank_calibrator_on_val(
    z0_val: np.ndarray,
    y_val: np.ndarray,
    delta_val: np.ndarray,
    alpha_val_raw: np.ndarray,
    alpha_global: float,
) -> dict:
    best = {
        "n_bins": 0,
        "gamma": 0.0,
        "edges": None,
        "bin_alphas": None,
        "val_error": error_for_alpha(z0_val, y_val, delta_val, alpha_global),
        "alpha_mean": alpha_global,
        "alpha_std": 0.0,
    }

    for n_bins in [2, 3, 4, 5, 8, 10]:
        edges, bin_alphas = fit_bin_alphas(z0_val, y_val, delta_val, alpha_val_raw, n_bins)
        bins_val = apply_bins(alpha_val_raw, edges)
        alpha_bin_val = bin_alphas[bins_val]

        for gamma in np.linspace(0.0, 1.0, 101):
            alpha = np.clip((1.0 - gamma) * alpha_global + gamma * alpha_bin_val, 0.0, 1.0)
            err = error_for_alpha(z0_val, y_val, delta_val, alpha)

            if err < best["val_error"]:
                best = {
                    "n_bins": int(n_bins),
                    "gamma": float(gamma),
                    "edges": edges,
                    "bin_alphas": bin_alphas,
                    "val_error": float(err),
                    "alpha_mean": float(alpha.mean()),
                    "alpha_std": float(alpha.std()),
                }

    return best


def apply_rank_calibrator(alpha_raw: np.ndarray, alpha_global: float, calib: dict) -> np.ndarray:
    if calib["n_bins"] == 0:
        return np.full_like(alpha_raw, fill_value=alpha_global, dtype=np.float32)

    bins = apply_bins(alpha_raw, calib["edges"])
    alpha_bin = calib["bin_alphas"][bins]
    alpha = np.clip((1.0 - calib["gamma"]) * alpha_global + calib["gamma"] * alpha_bin, 0.0, 1.0)
    return alpha.astype(np.float32)


def summarize_method(
    seed: int,
    split: str,
    method: str,
    z0: np.ndarray,
    y: np.ndarray,
    delta_hat: np.ndarray,
    alpha: float | np.ndarray,
    identity_error: float,
    extra: dict | None = None,
) -> dict:
    err = error_for_alpha(z0, y, delta_hat, alpha)

    if np.isscalar(alpha):
        alpha_mean = float(alpha)
        alpha_std = 0.0
    else:
        alpha_mean = float(np.mean(alpha))
        alpha_std = float(np.std(alpha))

    row = {
        "seed": seed,
        "split": split,
        "method": method,
        "error": err,
        "identity_error": identity_error,
        "improvement_vs_identity": identity_error - err,
        "ratio_vs_identity": err / identity_error,
        "alpha_mean": alpha_mean,
        "alpha_std": alpha_std,
    }

    if extra:
        row.update(extra)

    return row


def pm(s: pd.Series) -> str:
    return f"{s.mean():.6f} ± {s.std(ddof=1):.6f}"


def write_md(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "# Rank-calibrated alpha-bin summary",
        "",
        "The calibrator bins the conditional alpha score on the validation environment, learns one optimal alpha per bin, and selects a shrinkage strength toward the global alpha on validation.",
        "",
        "| split | method | seeds | error | identity error | improvement vs identity | ratio vs identity | alpha mean | alpha std | n bins | gamma |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for (split, method), g in df.groupby(["split", "method"], sort=False):
        n_bins = pm(g["n_bins"]) if "n_bins" in g else "0.000000 ± 0.000000"
        gamma = pm(g["gamma"]) if "gamma" in g else "0.000000 ± 0.000000"

        lines.append(
            f"| {split} | {method} | {len(g)} | "
            f"{pm(g['error'])} | {pm(g['identity_error'])} | "
            f"{pm(g['improvement_vs_identity'])} | {pm(g['ratio_vs_identity'])} | "
            f"{pm(g['alpha_mean'])} | {pm(g['alpha_std'])} | "
            f"{n_bins} | {gamma} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    from sklearn.ensemble import HistGradientBoostingRegressor

    train = load_npz(args.train)
    val = load_npz(args.val)
    test = load_npz(args.test)

    rows = []

    for seed in args.seeds:
        print("\n" + "=" * 100)
        print(f"Seed {seed}")

        ckpt = torch.load(Path(args.checkpoint_template.format(seed=seed)), map_location=args.device)
        stats = torch_stats_to_numpy(ckpt["stats"])
        model = build_model(ckpt, args.model, args.device)

        print("Predicting residuals...")
        d_train = predict_delta(model, normalized_input(train, stats), stats, args.batch_size, args.device)
        d_val = predict_delta(model, normalized_input(val, stats), stats, args.batch_size, args.device)
        d_test = predict_delta(model, normalized_input(test, stats), stats, args.batch_size, args.device)

        z0_val = val["z_current"].astype(np.float32)
        y_val = val["z_future"].astype(np.float32)
        z0_test = test["z_current"].astype(np.float32)
        y_test = test["z_future"].astype(np.float32)

        alpha_global, _ = select_global_alpha(z0_val, y_val, d_val)

        print("Training conditional alpha regressor...")
        y_alpha_train = oracle_alpha(
            d_train,
            train["z_future"].astype(np.float32) - train["z_current"].astype(np.float32),
        )

        alpha_model = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=1e-3,
            random_state=seed,
        )
        alpha_model.fit(compact_alpha_features(train, d_train), y_alpha_train)

        alpha_val_raw = np.clip(alpha_model.predict(compact_alpha_features(val, d_val)).astype(np.float32), 0.0, 1.0)
        alpha_test_raw = np.clip(alpha_model.predict(compact_alpha_features(test, d_test)).astype(np.float32), 0.0, 1.0)

        print("Selecting rank calibrator on validation...")
        calib = select_rank_calibrator_on_val(z0_val, y_val, d_val, alpha_val_raw, alpha_global)

        alpha_val_rank = apply_rank_calibrator(alpha_val_raw, alpha_global, calib)
        alpha_test_rank = apply_rank_calibrator(alpha_test_raw, alpha_global, calib)

        alpha_val_oracle = oracle_alpha(d_val, y_val - z0_val)
        alpha_test_oracle = oracle_alpha(d_test, y_test - z0_test)

        id_val = mean_mse(z0_val, y_val)
        id_test = mean_mse(z0_test, y_test)

        extra = {"n_bins": calib["n_bins"], "gamma": calib["gamma"]}

        for split, z0, y, d, id_err, a_rank, a_oracle in [
            ("val", z0_val, y_val, d_val, id_val, alpha_val_rank, alpha_val_oracle),
            ("test", z0_test, y_test, d_test, id_test, alpha_test_rank, alpha_test_oracle),
        ]:
            rows.append(summarize_method(seed, split, "identity", z0, y, d, 0.0, id_err, {"n_bins": 0, "gamma": 0.0}))
            rows.append(summarize_method(seed, split, "global_alpha_val", z0, y, d, alpha_global, id_err, {"n_bins": 0, "gamma": 0.0}))
            rows.append(summarize_method(seed, split, "rank_calibrated_alpha_bins", z0, y, d, a_rank, id_err, extra))
            rows.append(summarize_method(seed, split, "oracle_per_sample_alpha", z0, y, d, a_oracle, id_err, {"n_bins": 0, "gamma": 0.0}))

        print(f"selected n_bins={calib['n_bins']} gamma={calib['gamma']:.3f} val_error={calib['val_error']:.6f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    csv_path = args.out_dir / "rank_calibrated_alpha_bins_summary.csv"
    md_path = args.out_dir / "rank_calibrated_alpha_bins_summary.md"

    df.to_csv(csv_path, index=False)
    write_md(md_path, df)

    print(md_path)
    print(md_path.read_text())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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
    select_mixing_gamma,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", choices=["cheap_residual", "expensive_residual"], default="cheap_residual")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def fit_affine_alpha_on_val(
    z0: np.ndarray,
    y: np.ndarray,
    delta_hat: np.ndarray,
    alpha_raw: np.ndarray,
    reference_error: float,
) -> tuple[float, float, float]:
    true_delta = y - z0

    a_term = np.mean(true_delta * true_delta, axis=1)
    b_term = np.mean(true_delta * delta_hat, axis=1)
    c_term = np.mean(delta_hat * delta_hat, axis=1)

    best_a = 0.0
    best_b = 0.0
    best_error = reference_error

    for a in np.linspace(-0.5, 0.5, 101):
        for b in np.linspace(0.0, 1.5, 151):
            alpha = np.clip(a + b * alpha_raw, 0.0, 1.0)
            err = float(np.mean(a_term - 2.0 * alpha * b_term + alpha * alpha * c_term))
            if err < best_error:
                best_error = err
                best_a = float(a)
                best_b = float(b)

    return best_a, best_b, best_error


def error_with_alpha(z0: np.ndarray, y: np.ndarray, delta_hat: np.ndarray, alpha: np.ndarray | float) -> float:
    if np.isscalar(alpha):
        pred = z0 + float(alpha) * delta_hat
    else:
        pred = z0 + alpha[:, None].astype(np.float32) * delta_hat
    return mean_mse(pred, y)


def summarize_split(
    split_name: str,
    data: dict[str, np.ndarray],
    delta_hat: np.ndarray,
    alpha_global: float,
    alpha_cond_raw: np.ndarray,
    gamma: float,
    affine_a: float,
    affine_b: float,
) -> list[dict]:
    z0 = data["z_current"].astype(np.float32)
    y = data["z_future"].astype(np.float32)
    true_delta = y - z0

    identity_error = mean_mse(z0, y)

    alpha_raw = np.clip(alpha_cond_raw, 0.0, 1.0).astype(np.float32)
    alpha_mix = np.clip((1.0 - gamma) * alpha_global + gamma * alpha_raw, 0.0, 1.0).astype(np.float32)
    alpha_affine = np.clip(affine_a + affine_b * alpha_raw, 0.0, 1.0).astype(np.float32)
    alpha_oracle = oracle_alpha(delta_hat, true_delta)

    methods = [
        ("identity", 0.0),
        ("raw_residual_alpha_1", 1.0),
        ("global_alpha_val", alpha_global),
        ("conditional_alpha_raw", alpha_raw),
        ("conditional_alpha_mixed_with_global", alpha_mix),
        ("conditional_alpha_affine_calibrated", alpha_affine),
        ("oracle_per_sample_alpha", alpha_oracle),
    ]

    rows = []
    for method, alpha in methods:
        err = error_with_alpha(z0, y, delta_hat, alpha)

        if np.isscalar(alpha):
            alpha_mean = float(alpha)
            alpha_std = 0.0
        else:
            alpha_mean = float(np.mean(alpha))
            alpha_std = float(np.std(alpha))

        rows.append(
            {
                "split": split_name,
                "method": method,
                "error": err,
                "identity_error": identity_error,
                "improvement_vs_identity": identity_error - err,
                "ratio_vs_identity": err / identity_error,
                "alpha_mean": alpha_mean,
                "alpha_std": alpha_std,
            }
        )

    return rows


def write_md(path: Path, rows: list[dict], alpha_global: float, gamma: float, affine_a: float, affine_b: float) -> None:
    lines = [
        "# Conditional residual shrinkage with affine validation calibration",
        "",
        "Prediction family:",
        "",
        "`z_pred = z_current + alpha(x) * delta_hat`",
        "",
        f"Global alpha selected on validation: `{alpha_global:.4f}`.",
        f"Mixture coefficient selected on validation: `gamma={gamma:.4f}`.",
        f"Affine calibration selected on validation: `alpha_cal = clip({affine_a:.4f} + {affine_b:.4f} * alpha_raw, 0, 1)`.",
        "",
        "| split | method | error | identity error | improvement vs identity | ratio vs identity | alpha mean | alpha std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['split']} | {r['method']} | "
            f"{r['error']:.6f} | {r['identity_error']:.6f} | "
            f"{r['improvement_vs_identity']:.6f} | {r['ratio_vs_identity']:.6f} | "
            f"{r['alpha_mean']:.6f} | {r['alpha_std']:.6f} |"
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

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    stats = torch_stats_to_numpy(ckpt["stats"])
    model = build_model(ckpt, args.model, args.device)

    print("Predicting residuals...")
    d_train = predict_delta(model, normalized_input(train, stats), stats, args.batch_size, args.device)
    d_val = predict_delta(model, normalized_input(val, stats), stats, args.batch_size, args.device)
    d_test = predict_delta(model, normalized_input(test, stats), stats, args.batch_size, args.device)

    print("Selecting global alpha on validation...")
    alpha_global, val_global_error = select_global_alpha(
        val["z_current"].astype(np.float32),
        val["z_future"].astype(np.float32),
        d_val,
    )

    print("Training conditional alpha model on train per-sample oracle alpha...")
    y_alpha_train = oracle_alpha(
        d_train,
        train["z_future"].astype(np.float32) - train["z_current"].astype(np.float32),
    )

    xa_train = compact_alpha_features(train, d_train)
    xa_val = compact_alpha_features(val, d_val)
    xa_test = compact_alpha_features(test, d_test)

    alpha_model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        random_state=args.seed,
    )
    alpha_model.fit(xa_train, y_alpha_train)

    alpha_val_raw = np.clip(alpha_model.predict(xa_val).astype(np.float32), 0.0, 1.0)
    alpha_test_raw = np.clip(alpha_model.predict(xa_test).astype(np.float32), 0.0, 1.0)

    print("Selecting mixture gamma on validation...")
    gamma, _ = select_mixing_gamma(
        val["z_current"].astype(np.float32),
        val["z_future"].astype(np.float32),
        d_val,
        alpha_global,
        alpha_val_raw,
    )

    print("Selecting affine calibration on validation...")
    affine_a, affine_b, affine_val_error = fit_affine_alpha_on_val(
        val["z_current"].astype(np.float32),
        val["z_future"].astype(np.float32),
        d_val,
        alpha_val_raw,
        val_global_error,
    )

    print(f"global alpha={alpha_global:.4f}")
    print(f"gamma={gamma:.4f}")
    print(f"affine a={affine_a:.4f}, b={affine_b:.4f}, val_error={affine_val_error:.6f}")

    rows = []
    rows += summarize_split("val", val, d_val, alpha_global, alpha_val_raw, gamma, affine_a, affine_b)
    rows += summarize_split("test", test, d_test, alpha_global, alpha_test_raw, gamma, affine_a, affine_b)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.out_dir / f"{args.model}_conditional_shrinkage_affine_seed{args.seed}.csv"
    md_path = args.out_dir / f"{args.model}_conditional_shrinkage_affine_seed{args.seed}.md"

    write_csv(csv_path, rows)
    write_md(md_path, rows, alpha_global, gamma, affine_a, affine_b)

    print(md_path)
    print(md_path.read_text())


if __name__ == "__main__":
    main()

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
    compact_alpha_features,
    oracle_alpha,
    mean_mse,
    select_global_alpha,
)
from scripts.eval_conditional_residual_shrinkage_affine import fit_affine_alpha_on_val


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


def corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    den = np.sqrt(np.sum(x * x) * np.sum(y * y)) + 1e-12
    return float(np.sum(x * y) / den)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return corr(xr, yr)


def auc_score_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(labels)) < 2:
            return float("nan")
        return float(roc_auc_score(labels, scores))
    except Exception:
        return float("nan")


def mse_per_sample(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean((pred - target) ** 2, axis=1)


def eval_rows_for_split(
    *,
    seed: int,
    split: str,
    data: dict[str, np.ndarray],
    delta_hat: np.ndarray,
    alpha_raw: np.ndarray,
    alpha_global: float,
    affine_a: float,
    affine_b: float,
) -> tuple[list[dict], pd.DataFrame]:
    z0 = data["z_current"].astype(np.float32)
    y = data["z_future"].astype(np.float32)
    true_delta = y - z0

    alpha_raw = np.clip(alpha_raw, 0.0, 1.0).astype(np.float32)
    alpha_affine = np.clip(affine_a + affine_b * alpha_raw, 0.0, 1.0).astype(np.float32)
    alpha_oracle = oracle_alpha(delta_hat, true_delta)

    pred_global = z0 + alpha_global * delta_hat
    pred_raw = z0 + alpha_raw[:, None] * delta_hat
    pred_affine = z0 + alpha_affine[:, None] * delta_hat
    pred_oracle = z0 + alpha_oracle[:, None] * delta_hat

    err_identity = mse_per_sample(z0, y)
    err_global = mse_per_sample(pred_global, y)
    err_raw = mse_per_sample(pred_raw, y)
    err_affine = mse_per_sample(pred_affine, y)
    err_oracle = mse_per_sample(pred_oracle, y)

    q75 = np.quantile(alpha_oracle, 0.75)
    q50 = np.quantile(alpha_oracle, 0.50)

    rows = [
        {
            "seed": seed,
            "split": split,
            "alpha_global": alpha_global,
            "affine_a": affine_a,
            "affine_b": affine_b,
            "alpha_raw_mean": float(alpha_raw.mean()),
            "alpha_raw_std": float(alpha_raw.std()),
            "alpha_affine_mean": float(alpha_affine.mean()),
            "alpha_affine_std": float(alpha_affine.std()),
            "alpha_oracle_mean": float(alpha_oracle.mean()),
            "alpha_oracle_std": float(alpha_oracle.std()),
            "pearson_raw_oracle": corr(alpha_raw, alpha_oracle),
            "spearman_raw_oracle": spearman(alpha_raw, alpha_oracle),
            "pearson_affine_oracle": corr(alpha_affine, alpha_oracle),
            "spearman_affine_oracle": spearman(alpha_affine, alpha_oracle),
            "auc_top25_oracle_alpha_raw": auc_score_binary((alpha_oracle >= q75).astype(int), alpha_raw),
            "auc_top50_oracle_alpha_raw": auc_score_binary((alpha_oracle >= q50).astype(int), alpha_raw),
            "identity_error": float(err_identity.mean()),
            "global_error": float(err_global.mean()),
            "raw_cond_error": float(err_raw.mean()),
            "affine_cond_error": float(err_affine.mean()),
            "oracle_error": float(err_oracle.mean()),
            "global_improvement_vs_identity": float(err_identity.mean() - err_global.mean()),
            "affine_improvement_vs_identity": float(err_identity.mean() - err_affine.mean()),
            "affine_improvement_vs_global": float(err_global.mean() - err_affine.mean()),
            "oracle_improvement_vs_global": float(err_global.mean() - err_oracle.mean()),
        }
    ]

    # Quantile bins of predicted raw alpha.
    df = pd.DataFrame(
        {
            "seed": seed,
            "split": split,
            "alpha_raw": alpha_raw,
            "alpha_affine": alpha_affine,
            "alpha_oracle": alpha_oracle,
            "err_global": err_global,
            "err_affine": err_affine,
            "err_oracle": err_oracle,
        }
    )

    # qcut can fail if many duplicate values; rank first makes bins stable.
    df["bin"] = pd.qcut(df["alpha_raw"].rank(method="first"), q=10, labels=False)

    binned = (
        df.groupby(["seed", "split", "bin"], as_index=False)
        .agg(
            n=("alpha_raw", "size"),
            alpha_raw_mean=("alpha_raw", "mean"),
            alpha_affine_mean=("alpha_affine", "mean"),
            alpha_oracle_mean=("alpha_oracle", "mean"),
            global_error=("err_global", "mean"),
            affine_error=("err_affine", "mean"),
            oracle_error=("err_oracle", "mean"),
        )
    )
    binned["affine_improvement_vs_global"] = binned["global_error"] - binned["affine_error"]
    binned["oracle_improvement_vs_global"] = binned["global_error"] - binned["oracle_error"]

    return rows, binned


def pm(x: pd.Series) -> str:
    return f"{x.mean():.6f} ± {x.std(ddof=1):.6f}"


def write_summary_md(path: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# Conditional alpha ranking diagnostics",
        "",
        "This diagnostic measures whether the conditional alpha model ranks local oracle shrinkage coefficients.",
        "",
        "| split | metric | mean ± std over seeds |",
        "| --- | --- | ---: |",
    ]

    metrics = [
        "alpha_raw_mean",
        "alpha_affine_mean",
        "alpha_oracle_mean",
        "pearson_raw_oracle",
        "spearman_raw_oracle",
        "auc_top25_oracle_alpha_raw",
        "identity_error",
        "global_error",
        "affine_cond_error",
        "oracle_error",
        "global_improvement_vs_identity",
        "affine_improvement_vs_identity",
        "affine_improvement_vs_global",
        "oracle_improvement_vs_global",
    ]

    for split in ["val", "test"]:
        sub = summary[summary["split"] == split]
        for m in metrics:
            lines.append(f"| {split} | {m} | {pm(sub[m])} |")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    from sklearn.ensemble import HistGradientBoostingRegressor

    train = load_npz(args.train)
    val = load_npz(args.val)
    test = load_npz(args.test)

    all_rows = []
    all_bins = []

    for seed in args.seeds:
        print("\n" + "=" * 100)
        print(f"Seed {seed}")

        ckpt_path = Path(args.checkpoint_template.format(seed=seed))
        ckpt = torch.load(ckpt_path, map_location=args.device)

        stats = torch_stats_to_numpy(ckpt["stats"])
        model = build_model(ckpt, args.model, args.device)

        print("Predicting residuals...")
        d_train = predict_delta(model, normalized_input(train, stats), stats, args.batch_size, args.device)
        d_val = predict_delta(model, normalized_input(val, stats), stats, args.batch_size, args.device)
        d_test = predict_delta(model, normalized_input(test, stats), stats, args.batch_size, args.device)

        alpha_global, val_global_error = select_global_alpha(
            val["z_current"].astype(np.float32),
            val["z_future"].astype(np.float32),
            d_val,
        )

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

        alpha_val_raw = np.clip(
            alpha_model.predict(compact_alpha_features(val, d_val)).astype(np.float32),
            0.0,
            1.0,
        )
        alpha_test_raw = np.clip(
            alpha_model.predict(compact_alpha_features(test, d_test)).astype(np.float32),
            0.0,
            1.0,
        )

        affine_a, affine_b, _ = fit_affine_alpha_on_val(
            val["z_current"].astype(np.float32),
            val["z_future"].astype(np.float32),
            d_val,
            alpha_val_raw,
            val_global_error,
        )

        for split, data, delta, alpha_raw in [
            ("val", val, d_val, alpha_val_raw),
            ("test", test, d_test, alpha_test_raw),
        ]:
            rows, bins = eval_rows_for_split(
                seed=seed,
                split=split,
                data=data,
                delta_hat=delta,
                alpha_raw=alpha_raw,
                alpha_global=alpha_global,
                affine_a=affine_a,
                affine_b=affine_b,
            )
            all_rows.extend(rows)
            all_bins.append(bins)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(all_rows)
    binned = pd.concat(all_bins, ignore_index=True)

    summary_csv = args.out_dir / "conditional_alpha_ranking_summary.csv"
    binned_csv = args.out_dir / "conditional_alpha_ranking_bins.csv"
    summary_md = args.out_dir / "conditional_alpha_ranking_summary.md"

    summary.to_csv(summary_csv, index=False)
    binned.to_csv(binned_csv, index=False)
    write_summary_md(summary_md, summary)

    print(summary_md)
    print(summary_md.read_text())


if __name__ == "__main__":
    main()

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


def auc_top25(scores: np.ndarray, oracle: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    thr = np.quantile(oracle, 0.75)
    labels = (oracle >= thr).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def mse_pred(z0: np.ndarray, y: np.ndarray, delta: np.ndarray, alpha: float | np.ndarray) -> float:
    if np.isscalar(alpha):
        pred = z0 + float(alpha) * delta
    else:
        pred = z0 + alpha[:, None].astype(np.float32) * delta
    return mean_mse(pred, y)


def rich_alpha_features(data: dict[str, np.ndarray], delta_hat: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    z = data["z_current"].astype(np.float32)
    action = data["action"].astype(np.float32)
    d = z.shape[1]

    x_mean = stats["x_mean"].astype(np.float32)
    x_std = stats["x_std"].astype(np.float32)
    delta_std = stats["delta_std"].astype(np.float32)

    z_norm = (z - x_mean[:d][None, :]) / x_std[:d][None, :]
    a_norm = (action - x_mean[d:][None, :]) / x_std[d:][None, :]
    delta_norm = delta_hat / delta_std[None, :]

    compact = compact_alpha_features(data, delta_hat)

    feats = np.concatenate(
        [
            z_norm,
            a_norm,
            delta_norm,
            np.abs(delta_norm),
            z_norm * delta_norm,
            compact,
        ],
        axis=1,
    )

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_predict_compact_hgb(x_train, y_train, x_val, x_test, seed: int):
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        random_state=seed,
    )
    model.fit(x_train, y_train)

    return (
        np.clip(model.predict(x_val).astype(np.float32), 0.0, 1.0),
        np.clip(model.predict(x_test).astype(np.float32), 0.0, 1.0),
        "compact_hgb",
        {"ridge_alpha": np.nan},
    )


def fit_predict_rich_ridge(x_train, y_train, x_val, y_val_oracle, x_test, seed: int):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    best = None

    for alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0]:
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=alpha, random_state=seed),
        )
        model.fit(x_train, y_train)

        pred_val = np.clip(model.predict(x_val).astype(np.float32), 0.0, 1.0)
        # Select ridge strength by oracle-alpha MSE on validation.
        # This is a diagnostic: does a richer feature map predict local alpha at all?
        err = float(np.mean((pred_val - y_val_oracle) ** 2))

        if best is None or err < best["err"]:
            best = {
                "err": err,
                "alpha": alpha,
                "model": model,
                "pred_val": pred_val,
            }

    pred_test = np.clip(best["model"].predict(x_test).astype(np.float32), 0.0, 1.0)

    return (
        best["pred_val"],
        pred_test,
        "rich_ridge",
        {"ridge_alpha": float(best["alpha"])},
    )


def evaluate_alpha_method(
    *,
    seed: int,
    split: str,
    method: str,
    z0: np.ndarray,
    y: np.ndarray,
    delta: np.ndarray,
    alpha_raw: np.ndarray | float,
    alpha_oracle: np.ndarray,
    alpha_global: float,
    affine_a: float | None,
    affine_b: float | None,
    extra: dict,
) -> list[dict]:
    identity_error = mean_mse(z0, y)
    global_error = mse_pred(z0, y, delta, alpha_global)

    rows = []

    if np.isscalar(alpha_raw):
        alpha_eval = float(alpha_raw)
        alpha_mean = float(alpha_raw)
        alpha_std = 0.0
        pear = np.nan
        spear = np.nan
        auc = np.nan
    else:
        alpha_eval = np.clip(alpha_raw, 0.0, 1.0).astype(np.float32)
        alpha_mean = float(alpha_eval.mean())
        alpha_std = float(alpha_eval.std())
        pear = corr(alpha_eval, alpha_oracle)
        spear = spearman(alpha_eval, alpha_oracle)
        auc = auc_top25(alpha_eval, alpha_oracle)

    raw_error = mse_pred(z0, y, delta, alpha_eval)

    rows.append(
        {
            "seed": seed,
            "split": split,
            "method": method + "_raw",
            "error": raw_error,
            "identity_error": identity_error,
            "global_error": global_error,
            "improvement_vs_identity": identity_error - raw_error,
            "improvement_vs_global": global_error - raw_error,
            "ratio_vs_identity": raw_error / identity_error,
            "alpha_mean": alpha_mean,
            "alpha_std": alpha_std,
            "pearson_oracle": pear,
            "spearman_oracle": spear,
            "auc_top25_oracle": auc,
            **extra,
        }
    )

    if not np.isscalar(alpha_raw) and affine_a is not None and affine_b is not None:
        alpha_affine = np.clip(affine_a + affine_b * alpha_raw, 0.0, 1.0).astype(np.float32)
        affine_error = mse_pred(z0, y, delta, alpha_affine)

        rows.append(
            {
                "seed": seed,
                "split": split,
                "method": method + "_affine_val_calibrated",
                "error": affine_error,
                "identity_error": identity_error,
                "global_error": global_error,
                "improvement_vs_identity": identity_error - affine_error,
                "improvement_vs_global": global_error - affine_error,
                "ratio_vs_identity": affine_error / identity_error,
                "alpha_mean": float(alpha_affine.mean()),
                "alpha_std": float(alpha_affine.std()),
                "pearson_oracle": corr(alpha_affine, alpha_oracle),
                "spearman_oracle": spearman(alpha_affine, alpha_oracle),
                "auc_top25_oracle": auc_top25(alpha_affine, alpha_oracle),
                **extra,
            }
        )

    return rows


def pm(s: pd.Series) -> str:
    if s.isna().all():
        return "nan"
    return f"{s.mean():.6f} ± {s.std(ddof=1):.6f}"


def write_summary_md(path: Path, df: pd.DataFrame) -> None:
    lines = [
        "# Rich alpha feature diagnostic",
        "",
        "This diagnostic tests whether richer latent features improve prediction of the local oracle shrinkage coefficient.",
        "",
        "| split | method | seeds | error | improvement vs identity | improvement vs global | Spearman oracle | AUC top25 oracle | alpha mean | alpha std | ridge alpha |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for (split, method), g in df.groupby(["split", "method"], sort=False):
        lines.append(
            f"| {split} | {method} | {len(g)} | "
            f"{pm(g['error'])} | {pm(g['improvement_vs_identity'])} | {pm(g['improvement_vs_global'])} | "
            f"{pm(g['spearman_oracle'])} | {pm(g['auc_top25_oracle'])} | "
            f"{pm(g['alpha_mean'])} | {pm(g['alpha_std'])} | {pm(g['ridge_alpha'])} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    train = load_npz(args.train)
    val = load_npz(args.val)
    test = load_npz(args.test)

    rows = []

    for seed in args.seeds:
        print("\n" + "=" * 100)
        print(f"Seed {seed}")

        ckpt = torch.load(Path(args.checkpoint_template.format(seed=seed)), map_location=args.device)
        stats = torch_stats_to_numpy(ckpt["stats"])
        residual_model = build_model(ckpt, args.model, args.device)

        print("Predicting residuals...")
        d_train = predict_delta(residual_model, normalized_input(train, stats), stats, args.batch_size, args.device)
        d_val = predict_delta(residual_model, normalized_input(val, stats), stats, args.batch_size, args.device)
        d_test = predict_delta(residual_model, normalized_input(test, stats), stats, args.batch_size, args.device)

        z0_train = train["z_current"].astype(np.float32)
        y_train = train["z_future"].astype(np.float32)
        z0_val = val["z_current"].astype(np.float32)
        y_val = val["z_future"].astype(np.float32)
        z0_test = test["z_current"].astype(np.float32)
        y_test = test["z_future"].astype(np.float32)

        y_alpha_train = oracle_alpha(d_train, y_train - z0_train)
        y_alpha_val = oracle_alpha(d_val, y_val - z0_val)
        y_alpha_test = oracle_alpha(d_test, y_test - z0_test)

        alpha_global, val_global_error = select_global_alpha(z0_val, y_val, d_val)

        feature_jobs = []

        print("Training compact HGB alpha model...")
        feature_jobs.append(
            fit_predict_compact_hgb(
                compact_alpha_features(train, d_train),
                y_alpha_train,
                compact_alpha_features(val, d_val),
                compact_alpha_features(test, d_test),
                seed,
            )
        )

        print("Training rich Ridge alpha model...")
        feature_jobs.append(
            fit_predict_rich_ridge(
                rich_alpha_features(train, d_train, stats),
                y_alpha_train,
                rich_alpha_features(val, d_val, stats),
                y_alpha_val,
                rich_alpha_features(test, d_test, stats),
                seed,
            )
        )

        for alpha_val_raw, alpha_test_raw, method_name, extra in feature_jobs:
            affine_a, affine_b, _ = fit_affine_alpha_on_val(
                z0_val,
                y_val,
                d_val,
                alpha_val_raw,
                val_global_error,
            )

            print(
                f"{method_name}: affine a={affine_a:.4f}, b={affine_b:.4f}, "
                f"val raw mean={alpha_val_raw.mean():.4f}"
            )

            rows += evaluate_alpha_method(
                seed=seed,
                split="val",
                method=method_name,
                z0=z0_val,
                y=y_val,
                delta=d_val,
                alpha_raw=alpha_val_raw,
                alpha_oracle=y_alpha_val,
                alpha_global=alpha_global,
                affine_a=affine_a,
                affine_b=affine_b,
                extra=extra,
            )

            rows += evaluate_alpha_method(
                seed=seed,
                split="test",
                method=method_name,
                z0=z0_test,
                y=y_test,
                delta=d_test,
                alpha_raw=alpha_test_raw,
                alpha_oracle=y_alpha_test,
                alpha_global=alpha_global,
                affine_a=affine_a,
                affine_b=affine_b,
                extra=extra,
            )

        # Add references.
        for split, z0, y, d, alpha_or in [
            ("val", z0_val, y_val, d_val, y_alpha_val),
            ("test", z0_test, y_test, d_test, y_alpha_test),
        ]:
            identity_error = mean_mse(z0, y)
            global_error = mse_pred(z0, y, d, alpha_global)
            oracle_error = mse_pred(z0, y, d, alpha_or)

            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "method": "global_alpha_val",
                    "error": global_error,
                    "identity_error": identity_error,
                    "global_error": global_error,
                    "improvement_vs_identity": identity_error - global_error,
                    "improvement_vs_global": 0.0,
                    "ratio_vs_identity": global_error / identity_error,
                    "alpha_mean": alpha_global,
                    "alpha_std": 0.0,
                    "pearson_oracle": np.nan,
                    "spearman_oracle": np.nan,
                    "auc_top25_oracle": np.nan,
                    "ridge_alpha": np.nan,
                }
            )

            rows.append(
                {
                    "seed": seed,
                    "split": split,
                    "method": "oracle_per_sample_alpha",
                    "error": oracle_error,
                    "identity_error": identity_error,
                    "global_error": global_error,
                    "improvement_vs_identity": identity_error - oracle_error,
                    "improvement_vs_global": global_error - oracle_error,
                    "ratio_vs_identity": oracle_error / identity_error,
                    "alpha_mean": float(alpha_or.mean()),
                    "alpha_std": float(alpha_or.std()),
                    "pearson_oracle": 1.0,
                    "spearman_oracle": 1.0,
                    "auc_top25_oracle": 1.0,
                    "ridge_alpha": np.nan,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)

    csv_path = args.out_dir / "rich_alpha_feature_summary.csv"
    md_path = args.out_dir / "rich_alpha_feature_summary.md"

    df.to_csv(csv_path, index=False)
    write_summary_md(md_path, df)

    print(md_path)
    print(md_path.read_text())


if __name__ == "__main__":
    main()

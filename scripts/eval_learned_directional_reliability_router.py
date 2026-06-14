from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_conditional_residual_shrinkage import load_npz
from scripts.plot_latent_geometry_diagnostics import predict_delta_directional


def per_sample_mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean((a - b) ** 2, axis=1)


def safe_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def make_features(data: dict[str, np.ndarray], delta: np.ndarray) -> tuple[np.ndarray, list[str]]:
    z = data["z_current"].astype(np.float32)
    action = data["action"].astype(np.float32)
    gap = data["gap"].astype(np.float32)[:, None]

    action_norm = np.linalg.norm(action, axis=1, keepdims=True)
    pred_norm = np.linalg.norm(delta, axis=1, keepdims=True)

    z_norm = np.linalg.norm(z, axis=1, keepdims=True)
    z_mean = z.mean(axis=1, keepdims=True)
    z_std = z.std(axis=1, keepdims=True)

    d_mean = delta.mean(axis=1, keepdims=True)
    d_std = delta.std(axis=1, keepdims=True)
    d_abs_mean = np.abs(delta).mean(axis=1, keepdims=True)
    d_abs_max = np.abs(delta).max(axis=1, keepdims=True)

    z_delta_cos = safe_cos(z, delta)[:, None]
    pred_to_z_ratio = pred_norm / (z_norm + 1e-12)

    expected = data.get("expected_unreliable", np.zeros(len(z), dtype=np.float32)).astype(np.float32)[:, None]

    blocks = [
        gap,
        action,
        np.abs(action),
        action_norm,
        pred_norm,
        z_norm,
        z_mean,
        z_std,
        d_mean,
        d_std,
        d_abs_mean,
        d_abs_max,
        z_delta_cos,
        pred_to_z_ratio,
        expected,
    ]

    names = (
        ["gap"]
        + [f"action_{i}" for i in range(action.shape[1])]
        + [f"abs_action_{i}" for i in range(action.shape[1])]
        + [
            "action_norm",
            "pred_norm",
            "z_norm",
            "z_mean",
            "z_std",
            "delta_mean",
            "delta_std",
            "delta_abs_mean",
            "delta_abs_max",
            "z_delta_cos",
            "pred_to_z_ratio",
            "expected_unreliable",
        ]
    )

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, names


def compute_split(data, delta, alpha: float):
    z0 = data["z_current"].astype(np.float32)
    y = data["z_future"].astype(np.float32)
    pred = z0 + alpha * delta

    identity_err = per_sample_mse(z0, y)
    residual_err = per_sample_mse(pred, y)
    gain = identity_err - residual_err

    X, feature_names = make_features(data, delta)

    return {
        "X": X,
        "feature_names": feature_names,
        "identity_err": identity_err,
        "residual_err": residual_err,
        "gain": gain,
        "label": (gain > 0.0).astype(np.int32),
    }


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def eval_policy(split, selected: np.ndarray) -> dict[str, float]:
    policy_err = np.where(selected, split["residual_err"], split["identity_err"])
    gain = split["gain"]

    return {
        "error": float(np.mean(policy_err)),
        "identity_error": float(np.mean(split["identity_err"])),
        "gain": float(np.mean(split["identity_err"]) - np.mean(policy_err)),
        "selected": float(np.mean(selected)),
        "selected_mean_gain": float(np.mean(gain[selected])) if selected.any() else 0.0,
        "selected_positive_frac": float(np.mean(gain[selected] > 0.0)) if selected.any() else 0.0,
    }


def choose_threshold(scores: np.ndarray, split, min_selected: float, max_selected: float):
    thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))

    best = None

    for direction in [">=", "<="]:
        for thr in thresholds:
            if direction == ">=":
                selected = scores >= thr
            else:
                selected = scores <= thr

            frac = float(np.mean(selected))
            if frac < min_selected or frac > max_selected:
                continue

            r = eval_policy(split, selected)
            row = {
                "threshold": float(thr),
                "direction": direction,
                **r,
            }

            if best is None or row["gain"] > best["gain"]:
                best = row

    if best is None:
        selected = np.ones_like(scores, dtype=bool)
        best = {
            "threshold": float(np.min(scores)),
            "direction": ">=",
            **eval_policy(split, selected),
        }

    return best


def apply_threshold(scores: np.ndarray, threshold: float, direction: str) -> np.ndarray:
    if direction == ">=":
        return scores >= threshold
    if direction == "<=":
        return scores <= threshold
    raise ValueError(direction)


def row_to_md(row):
    return (
        f"| {row['policy']} | {row['score']} | {row['direction']} | {row['threshold']:.6f} | "
        f"{row['val_gain']:.6f} | {row['test_gain']:.6f} | {row['test_error']:.6f} | "
        f"{row['test_selected']:.6f} | {row['test_selected_mean_gain']:.6f} | "
        f"{row['test_selected_positive_frac']:.6f} | {row['val_auroc']:.6f} | {row['test_auroc']:.6f} |"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    p.add_argument("--min-selected", type=float, default=0.05)
    p.add_argument("--max-selected", type=float, default=1.00)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    print("Loading data...")
    train_data = load_npz(args.train)
    val_data = load_npz(args.val)
    test_data = load_npz(args.test)

    print("Predicting train residuals...")
    train_delta = predict_delta_directional(args.checkpoint, train_data, args.batch_size, args.device)
    print("Predicting val residuals...")
    val_delta = predict_delta_directional(args.checkpoint, val_data, args.batch_size, args.device)
    print("Predicting test residuals...")
    test_delta = predict_delta_directional(args.checkpoint, test_data, args.batch_size, args.device)

    metrics = json.loads((args.checkpoint.parent / "metrics.json").read_text())
    alpha = float(metrics["val"]["global_alpha"])

    train = compute_split(train_data, train_delta, alpha)
    val = compute_split(val_data, val_delta, alpha)
    test = compute_split(test_data, test_delta, alpha)

    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression, Ridge

    rows = []

    always = eval_policy(test, np.ones_like(test["label"], dtype=bool))
    rows.append({
        "policy": "always_residual",
        "score": "none",
        "direction": "-",
        "threshold": 0.0,
        "val_gain": eval_policy(val, np.ones_like(val["label"], dtype=bool))["gain"],
        "test_gain": always["gain"],
        "test_error": always["error"],
        "test_selected": always["selected"],
        "test_selected_mean_gain": always["selected_mean_gain"],
        "test_selected_positive_frac": always["selected_positive_frac"],
        "val_auroc": float("nan"),
        "test_auroc": float("nan"),
    })

    models = []

    clf_hgb = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        random_state=args.seed,
    )
    clf_hgb.fit(train["X"], train["label"])
    models.append(("HGB_classifier_proba", clf_hgb.predict_proba(val["X"])[:, 1], clf_hgb.predict_proba(test["X"])[:, 1]))

    reg_hgb = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        random_state=args.seed,
    )
    reg_hgb.fit(train["X"], train["gain"])
    models.append(("HGB_gain_regressor", reg_hgb.predict(val["X"]), reg_hgb.predict(test["X"])))

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.3, random_state=args.seed),
    )
    logreg.fit(train["X"], train["label"])
    models.append(("logistic_classifier_proba", logreg.predict_proba(val["X"])[:, 1], logreg.predict_proba(test["X"])[:, 1]))

    ridge = make_pipeline(
        StandardScaler(),
        Ridge(alpha=100.0, random_state=args.seed),
    )
    ridge.fit(train["X"], train["gain"])
    models.append(("ridge_gain_regressor", ridge.predict(val["X"]), ridge.predict(test["X"])))

    for name, val_score, test_score in models:
        best = choose_threshold(val_score, val, args.min_selected, args.max_selected)
        test_selected = apply_threshold(test_score, best["threshold"], best["direction"])
        test_eval = eval_policy(test, test_selected)

        rows.append({
            "policy": "learned_router",
            "score": name,
            "direction": best["direction"],
            "threshold": best["threshold"],
            "val_gain": best["gain"],
            "test_gain": test_eval["gain"],
            "test_error": test_eval["error"],
            "test_selected": test_eval["selected"],
            "test_selected_mean_gain": test_eval["selected_mean_gain"],
            "test_selected_positive_frac": test_eval["selected_positive_frac"],
            "val_auroc": auc_score(val["label"], val_score),
            "test_auroc": auc_score(test["label"], test_score),
        })

    oracle_selected = test["gain"] > 0.0
    oracle = eval_policy(test, oracle_selected)

    rows.append({
        "policy": "test_oracle_upper_bound",
        "score": "gain",
        "direction": ">0",
        "threshold": 0.0,
        "val_gain": float("nan"),
        "test_gain": oracle["gain"],
        "test_error": oracle["error"],
        "test_selected": oracle["selected"],
        "test_selected_mean_gain": oracle["selected_mean_gain"],
        "test_selected_positive_frac": oracle["selected_positive_frac"],
        "val_auroc": float("nan"),
        "test_auroc": 1.0,
    })

    lines = [
        "# Learned directional reliability router",
        "",
        f"Residual alpha selected on validation: `{alpha:.6f}`.",
        "",
        "The router is trained on train, thresholded on validation, and evaluated on the held-out test environment.",
        "",
        "| policy | score | direction | threshold | val gain | test gain | test error | test selected | selected mean gain | selected positive frac | val AUROC | test AUROC |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(row_to_md(r))

    out = args.out_dir / "learned_directional_reliability_router.md"
    out.write_text("\n".join(lines) + "\n")

    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()

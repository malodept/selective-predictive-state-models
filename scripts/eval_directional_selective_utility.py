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
from scripts.eval_learned_directional_reliability_router import (
    compute_split,
    make_features,
    auc_score,
)


def eval_policy(split, selected: np.ndarray, lam: float):
    policy_err = np.where(selected, split["residual_err"], split["identity_err"])
    gain = float(np.mean(split["identity_err"]) - np.mean(policy_err))
    selected_frac = float(np.mean(selected))
    utility = gain - lam * selected_frac

    return {
        "error": float(np.mean(policy_err)),
        "gain": gain,
        "selected": selected_frac,
        "utility": utility,
        "selected_mean_gain": float(np.mean(split["gain"][selected])) if selected.any() else 0.0,
        "selected_positive_frac": float(np.mean(split["gain"][selected] > 0.0)) if selected.any() else 0.0,
    }


def thresholds(scores: np.ndarray):
    return np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))


def choose_threshold_by_utility(scores: np.ndarray, split, lam: float):
    best = None

    for direction in [">=", "<="]:
        for thr in thresholds(scores):
            selected = scores >= thr if direction == ">=" else scores <= thr
            r = eval_policy(split, selected, lam)
            row = {
                "direction": direction,
                "threshold": float(thr),
                **r,
            }

            if best is None or row["utility"] > best["utility"]:
                best = row

    return best


def apply(scores, threshold: float, direction: str):
    if direction == ">=":
        return scores >= threshold
    if direction == "<=":
        return scores <= threshold
    raise ValueError(direction)


def row_md(r):
    return (
        f"| {r['lambda']:.4f} | {r['policy']} | {r['score']} | {r['direction']} | "
        f"{r['threshold']:.6f} | {r['val_utility']:.6f} | {r['test_utility']:.6f} | "
        f"{r['test_gain']:.6f} | {r['test_selected']:.6f} | "
        f"{r['test_selected_mean_gain']:.6f} | {r['test_positive_frac']:.6f} | "
        f"{r['val_auc']:.6f} | {r['test_auc']:.6f} |"
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
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

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
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, Ridge

    models = []

    hgb_clf = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        random_state=args.seed,
    )
    hgb_clf.fit(train["X"], train["label"])
    models.append(("HGB_classifier_proba", hgb_clf.predict_proba(val["X"])[:, 1], hgb_clf.predict_proba(test["X"])[:, 1]))

    hgb_reg = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=1e-3,
        random_state=args.seed,
    )
    hgb_reg.fit(train["X"], train["gain"])
    models.append(("HGB_gain_regressor", hgb_reg.predict(val["X"]), hgb_reg.predict(test["X"])))

    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, C=0.3, random_state=args.seed),
    )
    logreg.fit(train["X"], train["label"])
    models.append(("logistic_classifier_proba", logreg.predict_proba(val["X"])[:, 1], logreg.predict_proba(test["X"])[:, 1]))

    ridge = make_pipeline(
        StandardScaler(),
        Ridge(alpha=100.0),
    )
    ridge.fit(train["X"], train["gain"])
    models.append(("ridge_gain_regressor", ridge.predict(val["X"]), ridge.predict(test["X"])))

    # Add simple deployable signals too.
    simple_scores = [
        ("gap", val["X"][:, 0], test["X"][:, 0]),
        ("action_norm", val["X"][:, 15], test["X"][:, 15]),
        ("pred_norm", val["X"][:, 16], test["X"][:, 16]),
    ]

    all_scores = simple_scores + models

    lambdas = [0.0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03]

    rows = []

    for lam in lambdas:
        always_val = eval_policy(val, np.ones_like(val["label"], dtype=bool), lam)
        always_test = eval_policy(test, np.ones_like(test["label"], dtype=bool), lam)

        rows.append({
            "lambda": lam,
            "policy": "always_residual",
            "score": "none",
            "direction": "-",
            "threshold": 0.0,
            "val_utility": always_val["utility"],
            "test_utility": always_test["utility"],
            "test_gain": always_test["gain"],
            "test_selected": always_test["selected"],
            "test_selected_mean_gain": always_test["selected_mean_gain"],
            "test_positive_frac": always_test["selected_positive_frac"],
            "val_auc": float("nan"),
            "test_auc": float("nan"),
        })

        identity_val = eval_policy(val, np.zeros_like(val["label"], dtype=bool), lam)
        identity_test = eval_policy(test, np.zeros_like(test["label"], dtype=bool), lam)

        rows.append({
            "lambda": lam,
            "policy": "identity_only",
            "score": "none",
            "direction": "-",
            "threshold": 0.0,
            "val_utility": identity_val["utility"],
            "test_utility": identity_test["utility"],
            "test_gain": identity_test["gain"],
            "test_selected": identity_test["selected"],
            "test_selected_mean_gain": 0.0,
            "test_positive_frac": 0.0,
            "val_auc": float("nan"),
            "test_auc": float("nan"),
        })

        for score_name, val_score, test_score in all_scores:
            best = choose_threshold_by_utility(val_score, val, lam)
            selected_test = apply(test_score, best["threshold"], best["direction"])
            test_eval = eval_policy(test, selected_test, lam)

            rows.append({
                "lambda": lam,
                "policy": "threshold_router",
                "score": score_name,
                "direction": best["direction"],
                "threshold": best["threshold"],
                "val_utility": best["utility"],
                "test_utility": test_eval["utility"],
                "test_gain": test_eval["gain"],
                "test_selected": test_eval["selected"],
                "test_selected_mean_gain": test_eval["selected_mean_gain"],
                "test_positive_frac": test_eval["selected_positive_frac"],
                "val_auc": auc_score(val["label"], val_score),
                "test_auc": auc_score(test["label"], test_score),
            })

    # Best per lambda on validation.
    best_rows = []
    for lam in lambdas:
        group = [r for r in rows if abs(r["lambda"] - lam) < 1e-12]
        best = max(group, key=lambda r: r["val_utility"])
        best_rows.append(best)

    lines = [
        "# Directional selective compute utility",
        "",
        f"Residual alpha selected on validation: `{alpha:.6f}`.",
        "",
        "Utility is `gain - lambda * selected_fraction`. Thresholds are selected on validation and evaluated on held-out test.",
        "",
        "## Best policy selected on validation for each lambda",
        "",
        "| lambda | policy | score | direction | threshold | val utility | test utility | test gain | test selected | selected mean gain | selected positive frac | val AUROC | test AUROC |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in best_rows:
        lines.append(row_md(r))

    lines += [
        "",
        "## All policies",
        "",
        "| lambda | policy | score | direction | threshold | val utility | test utility | test gain | test selected | selected mean gain | selected positive frac | val AUROC | test AUROC |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(row_md(r))

    out = args.out_dir / "directional_selective_utility.md"
    out.write_text("\n".join(lines) + "\n")

    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def mean_std(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return float(values.mean()), 0.0
    return float(values.mean()), float(values.std(ddof=1))


def fmt(values):
    m, s = mean_std(values)
    return f"{m:.4f} ± {s:.4f}"


def utility(error, compute, lam):
    return -float(error) - lam * float(compute)


def eval_selection(cheap_err, exp_err, selected, lam, cheap_compute=1.0, exp_compute=4.0):
    selected = selected.astype(bool)
    err = np.where(selected, exp_err, cheap_err)
    mean_error = float(err.mean())
    selected_fraction = float(selected.mean())
    mean_compute = cheap_compute + selected_fraction * (exp_compute - cheap_compute)
    return {
        "error": mean_error,
        "compute": mean_compute,
        "selected": selected_fraction,
        "utility": utility(mean_error, mean_compute, lam),
    }


def best_threshold_from_score(score, cheap_err, exp_err, lam, maximize=True):
    """
    Higher score => route to expensive.
    Threshold is selected on calibration data.
    """
    score = np.asarray(score)
    qs = np.linspace(0.0, 1.0, 101)
    thresholds = np.unique(np.quantile(score, qs))
    thresholds = np.concatenate(
        [[score.max() + 1e-6], thresholds, [score.min() - 1e-6]]
    )

    best = None
    for tau in thresholds:
        selected = score >= tau
        row = eval_selection(cheap_err, exp_err, selected, lam)
        row["threshold"] = float(tau)
        if best is None or row["utility"] > best["utility"]:
            best = row
    return best


def zscore_train_apply(x_train, x_eval):
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (x_train - mu) / sigma, (x_eval - mu) / sigma


def fit_ridge(X, y, l2=10.0):
    """
    Tiny closed-form ridge regressor.
    The intercept is not regularized.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    X_aug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    reg = np.eye(X_aug.shape[1]) * l2
    reg[0, 0] = 0.0

    w = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ y)

    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float64)
        X_new_aug = np.concatenate([np.ones((X_new.shape[0], 1)), X_new], axis=1)
        return X_new_aug @ w

    return predict


def build_small_features(d):
    rel = d["reliability_scores"].reshape(-1, 1)
    action_norm = d["action_norm"].reshape(-1, 1)
    action = d["action"]
    return np.concatenate(
        [
            rel,
            action_norm,
            action,
            rel * action_norm,
        ],
        axis=1,
    )


def add_method(rows_by_method, method, seed_row):
    rows_by_method.setdefault(method, []).append(seed_row)


def plot_error_compute(summary, out_path):
    methods = list(summary.keys())

    x = []
    y = []
    xerr = []
    yerr = []
    labels = []

    for method in methods:
        rows = summary[method]
        em, es = mean_std([r["error"] for r in rows])
        cm, cs = mean_std([r["compute"] for r in rows])
        x.append(cm)
        y.append(em)
        xerr.append(cs)
        yerr.append(es)
        labels.append(method)

    plt.figure(figsize=(10, 6))
    plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=4)

    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=9)

    plt.title("Gain router and routing baselines: error--compute trade-off")
    plt.xlabel("mean compute")
    plt.ylabel("mean prediction error")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_delta_utility(summary, out_path):
    cheap_utils = [r["utility"] for r in summary["cheap-only"]]
    cheap_mean = np.mean(cheap_utils)

    methods = [m for m in summary.keys() if m != "oracle gain upper bound"]
    deltas = []
    errs = []

    for method in methods:
        vals = np.asarray([r["utility"] for r in summary[method]]) - np.asarray(cheap_utils)
        deltas.append(vals.mean())
        errs.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)

    plt.figure(figsize=(11, 5))
    xs = np.arange(len(methods))
    plt.bar(xs, deltas, yerr=errs, capsize=4)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks(xs, methods, rotation=25, ha="right")
    plt.ylabel("utility improvement over cheap-only")
    plt.title("Deployment utility gain at lambda=0.04")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-glob",
        default="outputs/bestval_dinov2_cheap10_exp80_seed*/eval_arrays.npz",
    )
    parser.add_argument("--lambda-compute", type=float, default=0.04)
    parser.add_argument("--cheap-compute", type=float, default=1.0)
    parser.add_argument("--expensive-compute", type=float, default=4.0)
    parser.add_argument("--output-table", default="reports/tables/gain_router_summary.md")
    parser.add_argument("--output-csv", default="reports/tables/gain_router_summary.csv")
    parser.add_argument("--figure-dir", default="reports/figures/gain_router")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--ridge-l2-small", type=float, default=1.0)
    parser.add_argument("--ridge-l2-state", type=float, default=50.0)
    args = parser.parse_args()

    run_paths = sorted(Path(".").glob(args.run_glob))
    if not run_paths:
        raise FileNotFoundError(f"No eval arrays found with glob: {args.run_glob}")

    lam = args.lambda_compute
    marginal_compute = args.expensive_compute - args.cheap_compute
    gain_threshold = lam * marginal_compute

    print("Using eval arrays:")
    for p in run_paths:
        print(" -", p)
    print(f"lambda={lam:.4f}")
    print(f"gain threshold = lambda * Δcompute = {gain_threshold:.4f}")

    rows_by_method = {}

    for seed_idx, p in enumerate(run_paths):
        d = np.load(p)

        cheap_err = d["cheap_errors"]
        exp_err = d["expensive_errors"]
        rel = d["reliability_scores"]
        action_norm = d["action_norm"]
        action = d["action"]
        val_x = d["val_x"]

        n = len(cheap_err)
        rng = np.random.default_rng(10_000 + seed_idx)
        perm = rng.permutation(n)

        n_cal = int(args.calibration_fraction * n)
        cal = perm[:n_cal]
        ev = perm[n_cal:]

        cheap_cal, cheap_ev = cheap_err[cal], cheap_err[ev]
        exp_cal, exp_ev = exp_err[cal], exp_err[ev]

        # Fixed policies.
        add_method(
            rows_by_method,
            "cheap-only",
            eval_selection(cheap_ev, exp_ev, np.zeros_like(cheap_ev, dtype=bool), lam),
        )
        add_method(
            rows_by_method,
            "all-expensive",
            eval_selection(cheap_ev, exp_ev, np.ones_like(cheap_ev, dtype=bool), lam),
        )

        # Learned reliability threshold.
        best_rel_cal = best_threshold_from_score(rel[cal], cheap_cal, exp_cal, lam)
        rel_selected_ev = rel[ev] >= best_rel_cal["threshold"]
        add_method(
            rows_by_method,
            "learned reliability",
            eval_selection(cheap_ev, exp_ev, rel_selected_ev, lam),
        )

        # Action norm threshold.
        best_action_cal = best_threshold_from_score(action_norm[cal], cheap_cal, exp_cal, lam)
        action_selected_ev = action_norm[ev] >= best_action_cal["threshold"]
        add_method(
            rows_by_method,
            "action norm",
            eval_selection(cheap_ev, exp_ev, action_selected_ev, lam),
        )

        # Hybrid reliability/action threshold.
        rel_cal_z, rel_ev_z = zscore_train_apply(rel[cal].reshape(-1, 1), rel[ev].reshape(-1, 1))
        act_cal_z, act_ev_z = zscore_train_apply(action_norm[cal].reshape(-1, 1), action_norm[ev].reshape(-1, 1))

        for alpha in [0.25, 0.50, 0.75]:
            hybrid_cal = (1.0 - alpha) * rel_cal_z[:, 0] + alpha * act_cal_z[:, 0]
            hybrid_ev = (1.0 - alpha) * rel_ev_z[:, 0] + alpha * act_ev_z[:, 0]
            best_hybrid_cal = best_threshold_from_score(hybrid_cal, cheap_cal, exp_cal, lam)
            selected_ev = hybrid_ev >= best_hybrid_cal["threshold"]
            add_method(
                rows_by_method,
                f"hybrid rel/action alpha={alpha:.2f}",
                eval_selection(cheap_ev, exp_ev, selected_ev, lam),
            )

        # Random same fraction as learned reliability.
        selected_fraction = float(rel_selected_ev.mean())
        random_selected = rng.random(len(ev)) < selected_fraction
        add_method(
            rows_by_method,
            "random same fraction",
            eval_selection(cheap_ev, exp_ev, random_selected, lam),
        )

        # Oracle gain upper bound.
        true_gain_ev = cheap_ev - exp_ev
        oracle_selected = true_gain_ev > gain_threshold
        add_method(
            rows_by_method,
            "oracle gain upper bound",
            eval_selection(cheap_ev, exp_ev, oracle_selected, lam),
        )

        # Learned gain router: small interpretable features.
        small = build_small_features(d)
        Xcal_small_raw = small[cal]
        Xev_small_raw = small[ev]
        Xcal_small, Xev_small = zscore_train_apply(Xcal_small_raw, Xev_small_raw)

        gain_cal = cheap_cal - exp_cal
        pred_gain_small = fit_ridge(Xcal_small, gain_cal, l2=args.ridge_l2_small)(Xev_small)

        gain_selected_theory = pred_gain_small > gain_threshold
        add_method(
            rows_by_method,
            "learned gain small, theoretical threshold",
            eval_selection(cheap_ev, exp_ev, gain_selected_theory, lam),
        )

        # Same learned gain score, but threshold calibrated for max utility.
        pred_gain_small_cal = fit_ridge(Xcal_small, gain_cal, l2=args.ridge_l2_small)(Xcal_small)
        best_gain_cal = best_threshold_from_score(pred_gain_small_cal, cheap_cal, exp_cal, lam)
        gain_selected_tuned = pred_gain_small >= best_gain_cal["threshold"]
        add_method(
            rows_by_method,
            "learned gain small, calibrated threshold",
            eval_selection(cheap_ev, exp_ev, gain_selected_tuned, lam),
        )

        # Learned gain router: full state-action features.
        Xcal_state_raw = val_x[cal]
        Xev_state_raw = val_x[ev]
        Xcal_state, Xev_state = zscore_train_apply(Xcal_state_raw, Xev_state_raw)

        pred_state_fn = fit_ridge(Xcal_state, gain_cal, l2=args.ridge_l2_state)
        pred_gain_state_cal = pred_state_fn(Xcal_state)
        pred_gain_state_ev = pred_state_fn(Xev_state)

        gain_state_theory = pred_gain_state_ev > gain_threshold
        add_method(
            rows_by_method,
            "learned gain state-action, theoretical threshold",
            eval_selection(cheap_ev, exp_ev, gain_state_theory, lam),
        )

        best_gain_state_cal = best_threshold_from_score(pred_gain_state_cal, cheap_cal, exp_cal, lam)
        gain_state_tuned = pred_gain_state_ev >= best_gain_state_cal["threshold"]
        add_method(
            rows_by_method,
            "learned gain state-action, calibrated threshold",
            eval_selection(cheap_ev, exp_ev, gain_state_tuned, lam),
        )

    out_table = Path(args.output_table)
    out_csv = Path(args.output_csv)
    fig_dir = Path(args.figure_dir)

    out_table.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    method_order = [
        "cheap-only",
        "all-expensive",
        "learned reliability",
        "action norm",
        "hybrid rel/action alpha=0.25",
        "hybrid rel/action alpha=0.50",
        "hybrid rel/action alpha=0.75",
        "random same fraction",
        "learned gain small, theoretical threshold",
        "learned gain small, calibrated threshold",
        "learned gain state-action, theoretical threshold",
        "learned gain state-action, calibrated threshold",
        "oracle gain upper bound",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "error_mean", "error_std", "compute_mean", "compute_std", "selected_mean", "selected_std", "utility_mean", "utility_std"])
        for method in method_order:
            if method not in rows_by_method:
                continue
            rows = rows_by_method[method]
            em, es = mean_std([r["error"] for r in rows])
            cm, cs = mean_std([r["compute"] for r in rows])
            sm, ss = mean_std([r["selected"] for r in rows])
            um, us = mean_std([r["utility"] for r in rows])
            writer.writerow([method, em, es, cm, cs, sm, ss, um, us])

    lines = []
    lines.append(f"# Gain-router baselines, DINOv2 best-validation, lambda={lam:.3f}\n")
    lines.append(f"Marginal compute threshold for using the expensive predictor: `{gain_threshold:.4f}`.\n")
    lines.append("| method | error | compute | selected | utility |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")

    for method in method_order:
        if method not in rows_by_method:
            continue
        rows = rows_by_method[method]
        lines.append(
            f"| {method} | "
            f"{fmt([r['error'] for r in rows])} | "
            f"{fmt([r['compute'] for r in rows])} | "
            f"{fmt([r['selected'] for r in rows])} | "
            f"{fmt([r['utility'] for r in rows])} |"
        )

    out_table.write_text("\n".join(lines) + "\n")

    plot_error_compute(
        {m: rows_by_method[m] for m in method_order if m in rows_by_method},
        fig_dir / "gain_router_error_compute.png",
    )
    plot_delta_utility(
        {m: rows_by_method[m] for m in method_order if m in rows_by_method},
        fig_dir / "gain_router_delta_utility.png",
    )

    print(out_table)
    print(out_csv)
    print(fig_dir / "gain_router_error_compute.png")
    print(fig_dir / "gain_router_delta_utility.png")
    print(out_table.read_text())


if __name__ == "__main__":
    main()

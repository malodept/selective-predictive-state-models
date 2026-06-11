from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from sklearn.ensemble import RandomForestClassifier
except Exception as exc:
    raise RuntimeError(
        "scikit-learn is required for this uncertainty diagnostic. "
        "If it is missing in the container, we will rewrite this with PyTorch."
    ) from exc

from scripts.trainval_gain_router import eval_arrays


COMPUTE_CHEAP = 1.0
COMPUTE_EXPENSIVE = 4.0
COMPUTE_GAP = COMPUTE_EXPENSIVE - COMPUTE_CHEAP


@dataclass
class Metrics:
    method: str
    error: float
    compute: float
    selected: float
    utility: float
    delta_utility: float


def get_array(d: Dict[str, np.ndarray], *names: str) -> np.ndarray:
    for name in names:
        if name in d:
            return np.asarray(d[name])
    raise KeyError(f"None of {names} found. Available keys: {sorted(d.keys())}")


def make_features(d: Dict[str, np.ndarray]) -> np.ndarray:
    x = get_array(d, "x", "val_x", "train_x", "inputs", "features").astype(np.float32)
    parts = [x]

    for names in [
        ("reliability_scores", "reliability", "score", "scores"),
        ("action_norm", "action_norms"),
    ]:
        try:
            arr = get_array(d, *names).astype(np.float32).reshape(len(x), -1)
            parts.append(arr)
        except KeyError:
            pass

    return np.concatenate(parts, axis=1).astype(np.float32)


def get_errors(d: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    cheap = get_array(d, "cheap_errors", "cheap_error").astype(np.float32).reshape(-1)
    expensive = get_array(d, "expensive_errors", "expensive_error").astype(np.float32).reshape(-1)
    return cheap, expensive


def standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def compute_metrics(
    method: str,
    cheap_error: np.ndarray,
    expensive_error: np.ndarray,
    selected: np.ndarray,
    lambda_compute: float,
    cheap_utility: float,
) -> Metrics:
    selected = selected.astype(bool)
    error = float(np.where(selected, expensive_error, cheap_error).mean())
    compute = float(COMPUTE_CHEAP + COMPUTE_GAP * selected.mean())
    utility = float(-error - lambda_compute * compute)
    return Metrics(
        method=method,
        error=error,
        compute=compute,
        selected=float(selected.mean()),
        utility=utility,
        delta_utility=float(utility - cheap_utility),
    )


def tree_probabilities(model: RandomForestClassifier, x: np.ndarray) -> np.ndarray:
    probs = []
    for tree in model.estimators_:
        p = tree.predict_proba(x)
        classes = tree.classes_
        idx = np.where(np.isclose(classes.astype(float), 1.0))[0]
        if len(idx) == 0:
            probs.append(np.zeros(len(x), dtype=np.float32))
        else:
            probs.append(p[:, idx[0]].astype(np.float32))
    return np.stack(probs, axis=0)


def best_threshold_for_score(
    score: np.ndarray,
    cheap_error: np.ndarray,
    expensive_error: np.ndarray,
    lambda_compute: float,
) -> float:
    qs = np.linspace(0.0, 1.0, 301)
    thresholds = np.unique(np.quantile(score, qs))
    thresholds = np.concatenate(
        [
            np.array([score.max() + 1e-6], dtype=np.float32),
            thresholds.astype(np.float32),
            np.array([score.min() - 1e-6], dtype=np.float32),
        ]
    )

    best_t = float(thresholds[0])
    best_u = -1e30

    for t in thresholds:
        selected = score >= t
        error = float(np.where(selected, expensive_error, cheap_error).mean())
        compute = float(COMPUTE_CHEAP + COMPUTE_GAP * selected.mean())
        utility = -error - lambda_compute * compute
        if utility > best_u:
            best_u = utility
            best_t = float(t)

    return best_t


def evaluate_seed(run_dir: Path, seed: int, args: argparse.Namespace) -> List[Metrics]:
    print(f" - seed={seed}: {run_dir}")

    train = eval_arrays(run_dir, "train", args.batch_size, args.device)
    val = eval_arrays(run_dir, "val", args.batch_size, args.device)

    x_train_raw = make_features(train)
    x_val_raw = make_features(val)

    cheap_train, expensive_train = get_errors(train)
    cheap_val, expensive_val = get_errors(val)

    true_gain_train = cheap_train - expensive_train
    true_gain_val = cheap_val - expensive_val
    marginal_threshold = args.lambda_compute * COMPUTE_GAP

    y_train_all = (true_gain_train > marginal_threshold).astype(np.int64)

    rng = np.random.default_rng(seed + 1234)
    idx = rng.permutation(len(x_train_raw))
    n_cal = max(512, int(args.calibration_fraction * len(idx)))
    cal_idx = idx[:n_cal]
    fit_idx = idx[n_cal:]

    x_fit_raw = x_train_raw[fit_idx]
    y_fit = y_train_all[fit_idx]

    x_cal_raw = x_train_raw[cal_idx]
    cheap_cal = cheap_train[cal_idx]
    expensive_cal = expensive_train[cal_idx]

    mean, std = standardize_fit(x_fit_raw)
    x_fit = standardize_apply(x_fit_raw, mean, std)
    x_cal = standardize_apply(x_cal_raw, mean, std)
    x_val = standardize_apply(x_val_raw, mean, std)

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(x_fit, y_fit)

    cal_tree_probs = tree_probabilities(clf, x_cal)
    val_tree_probs = tree_probabilities(clf, x_val)

    p_cal = cal_tree_probs.mean(axis=0)
    u_cal = cal_tree_probs.std(axis=0)
    p_val = val_tree_probs.mean(axis=0)
    u_val = val_tree_probs.std(axis=0)

    cheap_selected = np.zeros_like(cheap_val, dtype=bool)
    all_selected = np.ones_like(cheap_val, dtype=bool)
    oracle_selected = true_gain_val > marginal_threshold

    cheap_utility = compute_metrics(
        "cheap-only",
        cheap_val,
        expensive_val,
        cheap_selected,
        args.lambda_compute,
        0.0,
    ).utility

    rows = [
        compute_metrics("cheap-only", cheap_val, expensive_val, cheap_selected, args.lambda_compute, cheap_utility),
        compute_metrics("all-expensive", cheap_val, expensive_val, all_selected, args.lambda_compute, cheap_utility),
    ]

    t_mean = best_threshold_for_score(p_cal, cheap_cal, expensive_cal, args.lambda_compute)
    rows.append(
        compute_metrics(
            "RF mean router",
            cheap_val,
            expensive_val,
            p_val >= t_mean,
            args.lambda_compute,
            cheap_utility,
        )
    )

    for beta in args.betas:
        score_cal = p_cal - beta * u_cal
        score_val = p_val - beta * u_val
        t = best_threshold_for_score(score_cal, cheap_cal, expensive_cal, args.lambda_compute)
        rows.append(
            compute_metrics(
                f"RF LCB beta={beta:g}",
                cheap_val,
                expensive_val,
                score_val >= t,
                args.lambda_compute,
                cheap_utility,
            )
        )

    score_cal = p_cal - args.fallback_beta * u_cal
    score_val = p_val - args.fallback_beta * u_val
    t_fb = best_threshold_for_score(score_cal, cheap_cal, expensive_cal, args.lambda_compute)

    cal_cheap_utility = compute_metrics(
        "cal cheap",
        cheap_cal,
        expensive_cal,
        np.zeros_like(cheap_cal, dtype=bool),
        args.lambda_compute,
        0.0,
    ).utility
    cal_router = compute_metrics(
        "cal router",
        cheap_cal,
        expensive_cal,
        score_cal >= t_fb,
        args.lambda_compute,
        cal_cheap_utility,
    )

    if cal_router.delta_utility >= args.fallback_margin:
        fb_selected = score_val >= t_fb
    else:
        fb_selected = cheap_selected

    rows.append(
        compute_metrics(
            "RF LCB + fallback",
            cheap_val,
            expensive_val,
            fb_selected,
            args.lambda_compute,
            cheap_utility,
        )
    )

    rows.append(
        compute_metrics(
            "oracle upper bound",
            cheap_val,
            expensive_val,
            oracle_selected,
            args.lambda_compute,
            cheap_utility,
        )
    )

    return rows


def aggregate(rows_by_seed: List[List[Metrics]]):
    methods = [m.method for m in rows_by_seed[0]]
    agg = {}

    for method in methods:
        agg[method] = {}
        for field in ["error", "compute", "selected", "utility", "delta_utility"]:
            vals = []
            for rows in rows_by_seed:
                row_map = {m.method: m for m in rows}
                vals.append(getattr(row_map[method], field))
            vals = np.asarray(vals, dtype=np.float64)
            std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
            agg[method][field] = (float(vals.mean()), std)

    return methods, agg


def fmt(pair, signed: bool = False) -> str:
    mean, std = pair
    prefix = "+" if signed and mean >= 0 else ""
    return f"{prefix}{mean:.4f} ± {std:.4f}"


def write_table(path: Path, methods, agg, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Train-to-validation uncertainty-aware gain router",
        "",
        f"Lambda compute: `{args.lambda_compute:.4f}`. Marginal expensive threshold: `{args.lambda_compute * COMPUTE_GAP:.4f}`.",
        f"Random forest estimators: `{args.n_estimators}`. Calibration fraction: `{args.calibration_fraction:.2f}`.",
        "",
        "| method | error | compute | selected | utility | delta utility vs cheap-only |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for method in methods:
        a = agg[method]
        lines.append(
            f"| {method} | {fmt(a['error'])} | {fmt(a['compute'])} | {fmt(a['selected'])} | "
            f"{fmt(a['utility'])} | {fmt(a['delta_utility'], signed=True)} |"
        )

    path.write_text("\n".join(lines) + "\n")


def plot_delta(path: Path, methods, agg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    y = np.arange(len(methods))
    gains = np.array([agg[m]["delta_utility"][0] for m in methods])
    stds = np.array([agg[m]["delta_utility"][1] for m in methods])

    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(y, gains, xerr=stds, capsize=4)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlabel("mean utility improvement over cheap-only")
    ax.set_title("Uncertainty-aware gain router at lambda=0.04")
    ax.grid(True, alpha=0.25)

    for yi, g in zip(y, gains):
        ax.text(
            g + (0.004 if g >= 0 else -0.004),
            yi,
            f"{g:+.3f}",
            va="center",
            ha="left" if g >= 0 else "right",
        )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_error_compute(path: Path, methods, agg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5.8))

    for method in methods:
        error, error_std = agg[method]["error"]
        compute, compute_std = agg[method]["compute"]
        ax.errorbar(compute, error, xerr=compute_std, yerr=error_std, fmt="o", capsize=4)
        ax.annotate(method, (compute, error), xytext=(6, 4), textcoords="offset points")

    ax.set_xlabel("mean compute cost")
    ax.set_ylabel("mean prediction error")
    ax.set_title("Uncertainty-aware gain router: error--compute trade-off")
    ax.text(0.02, 0.05, "Lower-left is better.", transform=ax.transAxes, alpha=0.75)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-glob", default="outputs/bestval_dinov2_cheap10_exp80_seed*")
    parser.add_argument("--lambda-compute", type=float, default=0.04)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--fallback-beta", type=float, default=1.0)
    parser.add_argument("--fallback-margin", type=float, default=0.003)
    args = parser.parse_args()

    run_dirs = sorted(Path(".").glob(args.run_glob))
    run_dirs = [p for p in run_dirs if (p / "bestval_checkpoint.pt").exists()]

    if not run_dirs:
        raise FileNotFoundError(f"No run dirs found for glob: {args.run_glob}")

    print("Using checkpoints:")
    rows_by_seed = []

    for i, run_dir in enumerate(run_dirs):
        seed_text = "".join(ch for ch in run_dir.name.split("seed")[-1] if ch.isdigit())
        seed = int(seed_text) if seed_text else i
        rows_by_seed.append(evaluate_seed(run_dir, seed, args))

    methods, agg = aggregate(rows_by_seed)

    table_path = Path("reports/tables/trainval_uncertainty_gain_router_summary.md")
    fig_dir = Path("reports/figures/gain_router")
    delta_path = fig_dir / "trainval_uncertainty_gain_router_delta_utility.png"
    ec_path = fig_dir / "trainval_uncertainty_gain_router_error_compute.png"

    write_table(table_path, methods, agg, args)
    plot_delta(delta_path, methods, agg)
    plot_error_compute(ec_path, methods, agg)

    print(table_path)
    print(delta_path)
    print(ec_path)


if __name__ == "__main__":
    main()

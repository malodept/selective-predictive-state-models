from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.protocol_loo_learned_gain_router import (
    CHEAP_COMPUTE,
    COMPUTE_GAP,
    EXPENSIVE_COMPUTE,
    LAMBDA_COMPUTE,
    THRESHOLD,
    calibrate_threshold,
    eval_on_feature_file,
    load_models,
    parse_run_name,
    utility,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-glob", default="outputs/protocol_loo_P*_cheap*_exp20_seed0")
    p.add_argument("--trajectory-dir", type=Path, default=Path("outputs/tartanair_dinov2_features_v2/loo"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--sample-calib-fraction", type=float, default=0.25)
    p.add_argument("--fallback-margin", type=float, default=0.0)
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def trajectory_id(path: Path) -> str:
    m = re.search(r"loo_(P\d+)_val\.npz", path.name)
    if not m:
        raise ValueError(f"Cannot infer trajectory id from {path}")
    return m.group(1)


def fit_rf(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=args.trees,
        max_depth=10,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(x, y)
    return clf


def positive_proba(clf: RandomForestClassifier, x: np.ndarray) -> np.ndarray:
    proba = clf.predict_proba(x)
    idx = np.where(clf.classes_ == 1)[0]
    if len(idx) == 0:
        return np.zeros(x.shape[0], dtype=np.float32)
    return proba[:, int(idx[0])].astype(np.float32)


def concat(items: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.concatenate([it[key] for it in items], axis=0)


def eval_method(
    rows: list[dict[str, object]],
    heldout: str,
    run_name: str,
    method: str,
    outer: dict[str, np.ndarray],
    selected: np.ndarray,
    cheap_util: float,
    extra: dict[str, object] | None = None,
) -> None:
    error, compute, selected_frac, util = utility(
        outer["cheap_errors"],
        outer["expensive_errors"],
        selected,
    )
    row = {
        "heldout": heldout,
        "run": run_name,
        "method": method,
        "error": error,
        "compute": compute,
        "selected": selected_frac,
        "utility": util,
        "delta_utility": util - cheap_util,
    }
    if extra:
        row.update(extra)
    rows.append(row)


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def format_pm(values: list[float], digits: int = 4) -> str:
    m, s = mean_std(values)
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_summary(path: Path, rows: list[dict[str, object]], inner_rows: list[dict[str, object]]) -> None:
    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)

    order = [
        "cheap-only",
        "all-expensive",
        "sample RF",
        "sample RF + fallback",
        "trajectory-aware RF median",
        "trajectory-aware RF conservative",
        "trajectory-aware RF safe fallback",
        "oracle upper bound",
    ]

    lines = []
    lines.append("# Leave-one-trajectory-out trajectory-aware gain-router calibration\n")
    lines.append(f"Lambda compute: `{LAMBDA_COMPUTE:.4f}`. Marginal expensive threshold: `{THRESHOLD:.4f}`.\n")
    lines.append(
        "The trajectory-aware variants choose routing thresholds from inner held-out trajectories "
        "rather than from a random sample-level calibration split.\n"
    )
    lines.append("| method | error | compute | selected | utility | delta utility vs cheap-only |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")

    for method in order:
        vals = by_method.get(method, [])
        if not vals:
            continue
        lines.append(
            f"| {method} | "
            f"{format_pm([float(r['error']) for r in vals])} | "
            f"{format_pm([float(r['compute']) for r in vals])} | "
            f"{format_pm([float(r['selected']) for r in vals])} | "
            f"{format_pm([float(r['utility']) for r in vals])} | "
            f"{format_pm([float(r['delta_utility']) for r in vals])} |"
        )

    if inner_rows:
        min_deltas = [float(r["calib_delta"]) for r in inner_rows]
        lines.append("\n## Inner trajectory calibration folds\n")
        lines.append(f"Inner calibration folds: `{len(inner_rows)}`.")
        lines.append(f"Mean inner calibration ΔU: `{np.mean(min_deltas):.4f}`.")
        lines.append(f"Minimum inner calibration ΔU: `{np.min(min_deltas):.4f}`.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def plot_summary(rows: list[dict[str, object]], fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    order = [
        "cheap-only",
        "all-expensive",
        "sample RF",
        "sample RF + fallback",
        "trajectory-aware RF median",
        "trajectory-aware RF conservative",
        "trajectory-aware RF safe fallback",
        "oracle upper bound",
    ]
    short = {
        "cheap-only": "cheap-only",
        "all-expensive": "all-exp.",
        "sample RF": "sample RF",
        "sample RF + fallback": "sample RF + fb",
        "trajectory-aware RF median": "traj-aware median",
        "trajectory-aware RF conservative": "traj-aware cons.",
        "trajectory-aware RF safe fallback": "traj-aware safe",
        "oracle upper bound": "oracle",
    }

    by_method: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)

    methods = [m for m in order if m in by_method]

    # Delta utility bar plot.
    gains = []
    gain_stds = []
    labels = []
    for m in methods:
        vals = [float(r["delta_utility"]) for r in by_method[m]]
        mu, sd = mean_std(vals)
        gains.append(mu)
        gain_stds.append(sd)
        labels.append(short[m])

    y = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(y, gains, xerr=gain_stds, capsize=4)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("mean utility improvement over cheap-only")
    ax.set_title("Trajectory-aware gain-router calibration under LOO trajectory shift")

    for yi, g in zip(y, gains):
        ax.text(g + (0.003 if g >= 0 else -0.003), yi, f"{g:+.3f}", va="center")

    fig.tight_layout()
    fig.savefig(fig_dir / "loo_trajectory_aware_gain_router_delta_utility_seed0.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Error-compute plot.
    fig, ax = plt.subplots(figsize=(10, 6))
    for m in methods:
        vals = by_method[m]
        xm, xs = mean_std([float(r["compute"]) for r in vals])
        ym, ys = mean_std([float(r["error"]) for r in vals])
        ax.errorbar(xm, ym, xerr=xs, yerr=ys, fmt="o", capsize=4)
        ax.annotate(short[m], xy=(xm, ym), xytext=(6, 4), textcoords="offset points")

    ax.set_xlabel("mean compute cost")
    ax.set_ylabel("mean prediction error")
    ax.set_title("Trajectory-aware gain-router: error--compute trade-off")
    ax.text(0.02, 0.04, "Lower-left is better.", transform=ax.transAxes, alpha=0.7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "loo_trajectory_aware_gain_router_error_compute_seed0.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_dirs = sorted(Path(".").glob(args.run_glob))
    if not run_dirs:
        raise FileNotFoundError(f"No runs found for glob: {args.run_glob}")

    traj_paths = {
        trajectory_id(p): p
        for p in sorted(args.trajectory_dir.glob("loo_P*_val.npz"))
    }
    if not traj_paths:
        raise FileNotFoundError(f"No trajectory files found in {args.trajectory_dir}")

    print("Trajectory files:")
    for tid, p in traj_paths.items():
        print(f" - {tid}: {p}")

    rows: list[dict[str, object]] = []
    inner_rows: list[dict[str, object]] = []

    for run_dir in run_dirs:
        heldout, run_name = parse_run_name(run_dir)
        print("\n" + "=" * 100)
        print(f"Run: {run_dir}")
        print(f"Heldout={heldout} run={run_name}")

        _, cheap, expensive, reliability = load_models(run_dir, args.device)

        eval_by_tid = {}
        for tid, path in traj_paths.items():
            eval_by_tid[tid] = eval_on_feature_file(path, cheap, expensive, reliability, args.batch_size, args.device)

        outer = eval_by_tid[heldout]
        train_tids = [tid for tid in sorted(eval_by_tid) if tid != heldout]
        train_items = [eval_by_tid[tid] for tid in train_tids]

        cheap_selected = np.zeros_like(outer["cheap_errors"], dtype=bool)
        expensive_selected = np.ones_like(outer["cheap_errors"], dtype=bool)
        oracle_selected = outer["gain"] > THRESHOLD

        cheap_util = utility(outer["cheap_errors"], outer["expensive_errors"], cheap_selected)[3]

        eval_method(rows, heldout, run_name, "cheap-only", outer, cheap_selected, cheap_util)
        eval_method(rows, heldout, run_name, "all-expensive", outer, expensive_selected, cheap_util)
        eval_method(rows, heldout, run_name, "oracle upper bound", outer, oracle_selected, cheap_util)

        train_x = concat(train_items, "router_x")
        train_y = concat(train_items, "profitable").astype(int)
        train_cheap = concat(train_items, "cheap_errors")
        train_exp = concat(train_items, "expensive_errors")

        # Baseline: random sample-level calibration, close to the previous diagnostic.
        idx = np.arange(len(train_y))
        stratify = train_y if len(np.unique(train_y)) == 2 else None
        fit_idx, calib_idx = train_test_split(
            idx,
            test_size=args.sample_calib_fraction,
            random_state=args.seed,
            stratify=stratify,
        )

        sample_clf = fit_rf(train_x[fit_idx], train_y[fit_idx], args)
        sample_calib_scores = positive_proba(sample_clf, train_x[calib_idx])
        sample_thr, sample_delta, sample_selected = calibrate_threshold(
            sample_calib_scores,
            train_cheap[calib_idx],
            train_exp[calib_idx],
        )

        outer_sample_scores = positive_proba(sample_clf, outer["router_x"])
        sample_sel = outer_sample_scores >= sample_thr
        sample_fb_sel = sample_sel if sample_delta > args.fallback_margin else cheap_selected

        eval_method(
            rows,
            heldout,
            run_name,
            "sample RF",
            outer,
            sample_sel,
            cheap_util,
            {"threshold": sample_thr, "calib_delta": sample_delta, "calib_selected": sample_selected},
        )
        eval_method(
            rows,
            heldout,
            run_name,
            "sample RF + fallback",
            outer,
            sample_fb_sel,
            cheap_util,
            {"threshold": sample_thr, "calib_delta": sample_delta, "calib_selected": sample_selected},
        )

        # Trajectory-aware inner calibration.
        fold_thresholds = []
        fold_deltas = []
        fold_selected = []

        for calib_tid in train_tids:
            fit_tids = [tid for tid in train_tids if tid != calib_tid]
            fit_items = [eval_by_tid[tid] for tid in fit_tids]
            calib = eval_by_tid[calib_tid]

            fit_x = concat(fit_items, "router_x")
            fit_y = concat(fit_items, "profitable").astype(int)

            clf = fit_rf(fit_x, fit_y, args)
            calib_scores = positive_proba(clf, calib["router_x"])

            thr, delta, selected_frac = calibrate_threshold(
                calib_scores,
                calib["cheap_errors"],
                calib["expensive_errors"],
            )

            fold_thresholds.append(thr)
            fold_deltas.append(delta)
            fold_selected.append(selected_frac)

            inner_rows.append(
                {
                    "outer_heldout": heldout,
                    "run": run_name,
                    "inner_calib": calib_tid,
                    "threshold": thr,
                    "calib_delta": delta,
                    "calib_selected": selected_frac,
                    "profitable_frac": float(calib["profitable"].mean()),
                }
            )

        median_thr = float(np.median(fold_thresholds))
        conservative_thr = float(np.max(fold_thresholds))
        min_inner_delta = float(np.min(fold_deltas))
        mean_inner_delta = float(np.mean(fold_deltas))

        final_clf = fit_rf(train_x, train_y, args)
        outer_scores = positive_proba(final_clf, outer["router_x"])

        median_sel = outer_scores >= median_thr
        conservative_sel = outer_scores >= conservative_thr
        safe_sel = conservative_sel if min_inner_delta > args.fallback_margin else cheap_selected

        common_extra = {
            "inner_min_delta": min_inner_delta,
            "inner_mean_delta": mean_inner_delta,
            "inner_median_threshold": median_thr,
            "inner_conservative_threshold": conservative_thr,
        }

        eval_method(
            rows,
            heldout,
            run_name,
            "trajectory-aware RF median",
            outer,
            median_sel,
            cheap_util,
            {**common_extra, "threshold": median_thr},
        )
        eval_method(
            rows,
            heldout,
            run_name,
            "trajectory-aware RF conservative",
            outer,
            conservative_sel,
            cheap_util,
            {**common_extra, "threshold": conservative_thr},
        )
        eval_method(
            rows,
            heldout,
            run_name,
            "trajectory-aware RF safe fallback",
            outer,
            safe_sel,
            cheap_util,
            {**common_extra, "threshold": conservative_thr},
        )

        print(f"sample calib delta={sample_delta:+.4f}, thr={sample_thr:.4f}")
        print(f"inner min delta={min_inner_delta:+.4f}, mean delta={mean_inner_delta:+.4f}")
        print(f"median_thr={median_thr:.4f}, conservative_thr={conservative_thr:.4f}")

    table_dir = Path("reports/tables/protocol")
    fig_dir = Path("reports/figures/protocol")

    write_csv(table_dir / "loo_trajectory_aware_gain_router_detailed_seed0.csv", rows)
    write_csv(table_dir / "loo_trajectory_aware_gain_router_inner_folds_seed0.csv", inner_rows)
    write_summary(table_dir / "loo_trajectory_aware_gain_router_summary_seed0.md", rows, inner_rows)
    plot_summary(rows, fig_dir)

    print(table_dir / "loo_trajectory_aware_gain_router_summary_seed0.md")
    print(table_dir / "loo_trajectory_aware_gain_router_detailed_seed0.csv")
    print(table_dir / "loo_trajectory_aware_gain_router_inner_folds_seed0.csv")
    print(fig_dir / "loo_trajectory_aware_gain_router_delta_utility_seed0.png")
    print(fig_dir / "loo_trajectory_aware_gain_router_error_compute_seed0.png")


if __name__ == "__main__":
    main()

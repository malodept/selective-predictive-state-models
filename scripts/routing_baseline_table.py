from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUNS = [
    Path("outputs/bestval_dinov2_cheap10_exp80_seed0/eval_arrays.npz"),
    Path("outputs/bestval_dinov2_cheap10_exp80_seed1/eval_arrays.npz"),
    Path("outputs/bestval_dinov2_cheap10_exp80_seed2/eval_arrays.npz"),
]

LAMBDA_COMPUTE = 0.04
CHEAP_COMPUTE = 1.0
EXPENSIVE_COMPUTE = 4.0
COMPUTE_DELTA = EXPENSIVE_COMPUTE - CHEAP_COMPUTE

TABLE_DIR = Path("reports/tables")
FIG_DIR = Path("reports/figures/routing_baselines")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def utility(error: float, compute: float) -> float:
    return -error - LAMBDA_COMPUTE * compute


def evaluate_selection(
    name: str,
    selected: np.ndarray,
    cheap_errors: np.ndarray,
    expensive_errors: np.ndarray,
) -> dict:
    selected = selected.astype(bool)
    p = float(selected.mean())
    errors = np.where(selected, expensive_errors, cheap_errors)
    err = float(errors.mean())
    compute = float(CHEAP_COMPUTE + COMPUTE_DELTA * p)
    return {
        "method": name,
        "error": err,
        "compute": compute,
        "selected": p,
        "utility": utility(err, compute),
    }


def best_topk_policy(
    name: str,
    score: np.ndarray,
    cheap_errors: np.ndarray,
    expensive_errors: np.ndarray,
) -> dict:
    n = len(score)
    order = np.argsort(-score)

    cheap_sum_total = float(cheap_errors.sum())

    # k = number of samples routed to expensive.
    best = evaluate_selection(
        name,
        np.zeros(n, dtype=bool),
        cheap_errors,
        expensive_errors,
    )
    best["k"] = 0

    selected = np.zeros(n, dtype=bool)

    # Incrementally route the top-k scored samples to expensive.
    current_error_sum = cheap_sum_total

    for k, idx in enumerate(order, start=1):
        current_error_sum += float(expensive_errors[idx] - cheap_errors[idx])
        selected[idx] = True

        p = k / n
        err = current_error_sum / n
        compute = CHEAP_COMPUTE + COMPUTE_DELTA * p
        u = utility(err, compute)

        if u > best["utility"]:
            best = {
                "method": name,
                "error": float(err),
                "compute": float(compute),
                "selected": float(p),
                "utility": float(u),
                "k": int(k),
            }

    return best


def minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    lo = float(x.min())
    hi = float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def random_same_fraction(
    learned: dict,
    cheap_errors: np.ndarray,
    expensive_errors: np.ndarray,
) -> dict:
    # Expected random routing at the same selected fraction as learned routing.
    p = float(learned["selected"])
    err = float((1.0 - p) * cheap_errors.mean() + p * expensive_errors.mean())
    compute = float(CHEAP_COMPUTE + COMPUTE_DELTA * p)
    return {
        "method": "random same fraction",
        "error": err,
        "compute": compute,
        "selected": p,
        "utility": utility(err, compute),
    }


def summarize(rows: list[dict]) -> list[dict]:
    methods = []
    for r in rows:
        if r["method"] not in methods:
            methods.append(r["method"])

    out = []
    for method in methods:
        subset = [r for r in rows if r["method"] == method]
        row = {"method": method}

        for key in ["error", "compute", "selected", "utility"]:
            vals = [float(r[key]) for r in subset]
            row[f"{key}_mean"] = mean(vals)
            row[f"{key}_std"] = stdev(vals) if len(vals) > 1 else 0.0

        out.append(row)

    return out


def fmt(m: float, s: float) -> str:
    return f"{m:.4f} ± {s:.4f}"


def main() -> None:
    per_seed = []

    for seed, path in enumerate(RUNS):
        if not path.exists():
            raise FileNotFoundError(path)

        d = np.load(path)

        cheap_errors = d["cheap_errors"].astype(np.float64)
        expensive_errors = d["expensive_errors"].astype(np.float64)
        reliability_scores = d["reliability_scores"].astype(np.float64)
        action_norm = d["action_norm"].astype(np.float64)

        gain = cheap_errors - expensive_errors

        cheap = evaluate_selection(
            "cheap-only",
            np.zeros_like(cheap_errors, dtype=bool),
            cheap_errors,
            expensive_errors,
        )
        expensive = evaluate_selection(
            "all-expensive",
            np.ones_like(cheap_errors, dtype=bool),
            cheap_errors,
            expensive_errors,
        )
        learned = best_topk_policy(
            "learned reliability",
            reliability_scores,
            cheap_errors,
            expensive_errors,
        )
        action = best_topk_policy(
            "action norm",
            action_norm,
            cheap_errors,
            expensive_errors,
        )

        reliability_z = minmax(reliability_scores)
        action_z = minmax(action_norm)

        hybrid_rows = []
        for alpha in [0.25, 0.50, 0.75]:
            hybrid_score = alpha * reliability_z + (1.0 - alpha) * action_z
            hybrid_rows.append(
                best_topk_policy(
                    f"hybrid rel/action alpha={alpha:.2f}",
                    hybrid_score,
                    cheap_errors,
                    expensive_errors,
                )
            )

        random_same = random_same_fraction(
            learned,
            cheap_errors,
            expensive_errors,
        )
        oracle = best_topk_policy(
            "oracle gain upper bound",
            gain,
            cheap_errors,
            expensive_errors,
        )

        for row in [cheap, expensive, learned, action, *hybrid_rows, random_same, oracle]:
            row = dict(row)
            row["seed"] = seed
            per_seed.append(row)

    summary = summarize(per_seed)

    # Write per-seed CSV.
    per_seed_csv = TABLE_DIR / "routing_baselines_per_seed.csv"
    with per_seed_csv.open("w", newline="") as f:
        fieldnames = ["seed", "method", "error", "compute", "selected", "utility"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_seed:
            writer.writerow({k: r.get(k) for k in fieldnames})

    # Write summary CSV.
    summary_csv = TABLE_DIR / "routing_baselines_summary.csv"
    with summary_csv.open("w", newline="") as f:
        fieldnames = list(summary[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    # Write markdown table.
    md = TABLE_DIR / "routing_baselines_summary.md"
    lines = []
    lines.append("# Routing baselines, DINOv2 best-validation, lambda=0.04\n\n")
    lines.append("| method | error | compute | selected | utility |\n")
    lines.append("| --- | ---: | ---: | ---: | ---: |\n")

    for r in summary:
        lines.append(
            f"| {r['method']} | "
            f"{fmt(r['error_mean'], r['error_std'])} | "
            f"{fmt(r['compute_mean'], r['compute_std'])} | "
            f"{fmt(r['selected_mean'], r['selected_std'])} | "
            f"{fmt(r['utility_mean'], r['utility_std'])} |\n"
        )

    md.write_text("".join(lines))
    print(md)
    print(md.read_text())

    # Figure: utility comparison.
    methods = [r["method"] for r in summary]
    utility_means = [r["utility_mean"] for r in summary]
    utility_stds = [r["utility_std"] for r in summary]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(methods))
    plt.bar(x, utility_means, yerr=utility_stds, capsize=4)
    plt.xticks(x, methods, rotation=25, ha="right")
    plt.ylabel("utility")
    plt.title("Routing baselines at lambda=0.04")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    out = FIG_DIR / "routing_baselines_utility.png"
    plt.savefig(out, dpi=220)
    plt.close()
    print(out)

    # Figure: error vs compute.
    plt.figure(figsize=(7, 5))
    for r in summary:
        plt.errorbar(
            r["compute_mean"],
            r["error_mean"],
            xerr=r["compute_std"],
            yerr=r["error_std"],
            marker="o",
            capsize=4,
            label=r["method"],
        )
    plt.xlabel("mean compute")
    plt.ylabel("mean prediction error")
    plt.title("Routing baselines: error--compute trade-off")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "routing_baselines_error_compute.png"
    plt.savefig(out, dpi=220)
    plt.close()
    print(out)


if __name__ == "__main__":
    main()

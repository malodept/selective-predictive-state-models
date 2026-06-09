from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_LAMBDAS = [
    0.000, 0.005, 0.010, 0.020, 0.030, 0.040, 0.050,
    0.075, 0.100, 0.150, 0.200, 0.250, 0.300,
]


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)


def find_selector_rows(obj):
    if isinstance(obj, dict):
        if isinstance(obj.get("selector_rows"), list):
            return obj["selector_rows"]
        if isinstance(obj.get("policies"), list):
            return obj["policies"]
        for value in obj.values():
            found = find_selector_rows(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_selector_rows(value)
            if found is not None:
                return found
    return None


def get_float(row: dict, keys: list[str]) -> float:
    for key in keys:
        if key in row and row[key] is not None:
            return float(row[key])
    raise KeyError(f"Could not find any of {keys} in row keys={sorted(row.keys())}")


def normalize_row(row: dict) -> dict:
    policy = str(row.get("policy", row.get("name", "")))
    if not policy:
        threshold = row.get("threshold")
        policy = "cheap-only" if threshold is None else f"threshold={threshold}"

    return {
        "policy": policy,
        "error": get_float(row, ["mean_error", "error", "best_error"]),
        "compute": get_float(row, ["mean_compute", "compute", "best_compute"]),
        "selected": float(row.get("selected_fraction", row.get("selected", 0.0))),
    }


def effective_policy_type(row: dict, eps: float = 0.01) -> str:
    """Classify the policy by its actual deployment behavior.

    Some threshold policies are mathematically distinct from the explicit
    baselines but behave almost identically in deployment. For example, a
    threshold that activates the expensive predictor on 99.95% of samples is
    effectively all-expensive. Collapsing these near-degenerate policies makes
    the regime plot easier to interpret.
    """
    selected = float(row.get("selected", row.get("selected_fraction", 0.0)))
    policy = str(row.get("policy", ""))

    if policy == "all-expensive" or selected >= 1.0 - eps:
        return "all-expensive"
    if policy == "cheap-only" or selected <= eps:
        return "cheap-only"
    return "adaptive"


def load_seed_run(path: Path) -> list[dict]:
    metrics = json.loads(path.read_text())
    rows = find_selector_rows(metrics)
    if rows is None:
        raise RuntimeError(f"No selector rows found in {path}")
    return [normalize_row(row) for row in rows]


def best_by_type(rows: list[dict], lam: float) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[effective_policy_type(row)].append(row)

    out = {}
    for typ, typ_rows in grouped.items():
        scored = []
        for row in typ_rows:
            utility = -row["error"] - lam * row["compute"]
            scored.append((utility, row))
        utility, row = max(scored, key=lambda x: x[0])
        out[typ] = {**row, "utility": utility}
    return out


def aggregate(records: list[dict], lambdas: list[float]) -> list[dict]:
    out = []

    for lam in lambdas:
        subset = [r for r in records if r["lambda"] == lam]
        best_errors = [r["best_error"] for r in subset]
        best_computes = [r["best_compute"] for r in subset]
        best_selected = [r["best_selected"] for r in subset]
        best_utilities = [r["best_utility"] for r in subset]

        policy_types = [r["best_type"] for r in subset]
        policy_names = [r["best_policy"] for r in subset]
        mode_type = Counter(policy_types).most_common(1)[0][0]

        row = {
            "lambda": lam,
            "policy_mode": mode_type,
            "policies": ", ".join(policy_names),
        }

        for prefix, values in [
            ("best_error", best_errors),
            ("best_compute", best_computes),
            ("best_selected", best_selected),
            ("best_utility", best_utilities),
        ]:
            m, s = mean_std(values)
            row[f"{prefix}_mean"] = m
            row[f"{prefix}_std"] = s

        for typ in ["cheap-only", "all-expensive", "adaptive"]:
            vals = [r[f"{typ}_utility"] for r in subset if f"{typ}_utility" in r]
            m, s = mean_std(vals)
            row[f"{typ}_utility_mean"] = m
            row[f"{typ}_utility_std"] = s

        out.append(row)

    return out


def fmt_pm(m: float, s: float, nd: int = 4) -> str:
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def write_markdown(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Best-validation DINOv2 real selective refinement: lambda sweep\n")
    lines.append("| lambda | policy mode | error | compute | selected | utility | policies |\n")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | --- |\n")

    for r in rows:
        lines.append(
            f"| {r['lambda']:.3f} | {r['policy_mode']} | "
            f"{fmt_pm(r['best_error_mean'], r['best_error_std'])} | "
            f"{fmt_pm(r['best_compute_mean'], r['best_compute_std'])} | "
            f"{fmt_pm(r['best_selected_mean'], r['best_selected_std'])} | "
            f"{fmt_pm(r['best_utility_mean'], r['best_utility_std'])} | "
            f"{r['policies']} |\n"
        )

    lines.append("\n## Utility by deployment policy\n\n")
    lines.append("| lambda | cheap-only | adaptive selector | all-expensive |\n")
    lines.append("| ---: | ---: | ---: | ---: |\n")
    for r in rows:
        lines.append(
            f"| {r['lambda']:.3f} | "
            f"{fmt_pm(r['cheap-only_utility_mean'], r['cheap-only_utility_std'])} | "
            f"{fmt_pm(r['adaptive_utility_mean'], r['adaptive_utility_std'])} | "
            f"{fmt_pm(r['all-expensive_utility_mean'], r['all-expensive_utility_std'])} |\n"
        )

    output.write_text("".join(lines))
    print(output)


def write_csv(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


def plot_summary(rows: list[dict], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    lam = [r["lambda"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Best-validation DINOv2 selective refinement across compute penalties", fontsize=18)

    ax = axes[0, 0]
    for typ, label in [
        ("cheap-only", "cheap-only"),
        ("adaptive", "adaptive selector"),
        ("all-expensive", "all-expensive"),
    ]:
        y = [r[f"{typ}_utility_mean"] for r in rows]
        ystd = [r[f"{typ}_utility_std"] for r in rows]
        ax.plot(lam, y, marker="o", label=label)
        ax.fill_between(lam, [a - b for a, b in zip(y, ystd)], [a + b for a, b in zip(y, ystd)], alpha=0.12)
    ax.set_title("Utility under deployment constraints")
    ax.set_xlabel("compute penalty λ")
    ax.set_ylabel("utility = -error - λ compute")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    y = [r["best_compute_mean"] for r in rows]
    ystd = [r["best_compute_std"] for r in rows]
    ax.errorbar(lam, y, yerr=ystd, marker="o", capsize=4)
    ax.set_title("Selected compute decreases as λ increases")
    ax.set_xlabel("compute penalty λ")
    ax.set_ylabel("mean compute cost")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    y = [r["best_selected_mean"] for r in rows]
    ystd = [r["best_selected_std"] for r in rows]
    ax.errorbar(lam, y, yerr=ystd, marker="o", capsize=4)
    ax.set_title("Expensive predictor activation")
    ax.set_xlabel("compute penalty λ")
    ax.set_ylabel("selected fraction")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    x = [r["best_compute_mean"] for r in rows]
    xstd = [r["best_compute_std"] for r in rows]
    y = [r["best_error_mean"] for r in rows]
    ystd = [r["best_error_std"] for r in rows]
    ax.errorbar(x, y, xerr=xstd, yerr=ystd, marker="o", capsize=4)
    for r in rows:
        if r["lambda"] in {0.0, 0.04, 0.15, 0.30}:
            ax.annotate(f"λ={r['lambda']:.2f}\n{r['policy_mode']}",
                        (r["best_compute_mean"], r["best_error_mean"]),
                        textcoords="offset points", xytext=(8, 8))
    ax.set_title("Error--compute frontier")
    ax.set_xlabel("mean compute cost")
    ax.set_ylabel("mean prediction error")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    path = figure_dir / "bestval_lambda_sweep_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(path)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    type_to_y = {"cheap-only": 0, "adaptive": 1, "all-expensive": 2}
    y = [type_to_y[r["policy_mode"]] for r in rows]
    ax.step(lam, y, where="mid", linewidth=2)
    ax.scatter(lam, y, s=70)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["cheap-only", "adaptive", "all-expensive"])
    ax.set_xlabel("compute penalty λ")
    ax.set_title("Best policy shifts with deployment constraints")
    ax.grid(alpha=0.25)
    path = figure_dir / "bestval_lambda_policy_regime.png"
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-glob",
        default="outputs/bestval_dinov2_cheap10_exp80_seed*/metrics.json",
        help="Glob for best-validation runs to reuse for the lambda sweep.",
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=DEFAULT_LAMBDAS,
    )
    parser.add_argument(
        "--table-out",
        default="reports/tables/bestval_lambda_sweep.md",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/tables/bestval_lambda_sweep.csv",
    )
    parser.add_argument(
        "--figure-dir",
        default="reports/figures/bestval_lambda_sweep",
    )
    args = parser.parse_args()

    paths = sorted(Path(".").glob(args.run_glob))
    if not paths:
        raise FileNotFoundError(f"No metrics files found with glob: {args.run_glob}")

    print("Using runs:")
    for p in paths:
        print(" -", p)

    records = []
    for seed, path in enumerate(paths):
        rows = load_seed_run(path)

        for lam in args.lambdas:
            candidates = []
            typed = best_by_type(rows, lam)

            for row in rows:
                utility = -row["error"] - lam * row["compute"]
                candidates.append((utility, row))

            best_utility, best = max(candidates, key=lambda x: x[0])
            rec = {
                "seed": seed,
                "lambda": lam,
                "best_policy": best["policy"],
                "best_type": effective_policy_type(best),
                "best_error": best["error"],
                "best_compute": best["compute"],
                "best_selected": best["selected"],
                "best_utility": best_utility,
            }

            for typ, row in typed.items():
                rec[f"{typ}_utility"] = row["utility"]

            records.append(rec)

    rows = aggregate(records, args.lambdas)

    write_markdown(rows, Path(args.table_out))
    write_csv(rows, Path(args.csv_out))
    plot_summary(rows, Path(args.figure_dir))


if __name__ == "__main__":
    main()

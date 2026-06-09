from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


CSV_PATH = Path("reports/tables/cheap_exp_gap_ablation_multiseed.csv")
FIG_DIR = Path("reports/figures/cheap_exp_gap_ablation")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_rows():
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in [
            "cheap_error_mean",
            "cheap_error_std",
            "expensive_error_mean",
            "expensive_error_std",
            "gap_mean",
            "gap_std",
            "best_error_mean",
            "best_error_std",
            "best_compute_mean",
            "best_compute_std",
            "selected_mean",
            "selected_std",
            "utility_mean",
            "utility_std",
        ]:
            r[k] = float(r[k])
    return rows


def short_label(run: str) -> str:
    return run.replace("/", "\n")


def main() -> None:
    rows = load_rows()

    names = [r["run"] for r in rows]
    x = list(range(len(rows)))

    # Figure 1: cheap/expensive gap and chosen policy.
    fig, ax = plt.subplots(figsize=(12, 7))

    gaps = [r["gap_mean"] for r in rows]
    gap_err = [r["gap_std"] for r in rows]

    ax.errorbar(x, gaps, yerr=gap_err, marker="o", linewidth=2, capsize=5)
    ax.axhline(0.0, linestyle="--", linewidth=1)

    for i, r in enumerate(rows):
        ax.text(
            i,
            r["gap_mean"] + 0.012,
            r["best_policy_mode"],
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([short_label(n) for n in names])
    ax.set_ylabel("cheap error - expensive error")
    ax.set_title("The useful selective-compute regime occurs at intermediate cheap/expensive gap")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out = FIG_DIR / "cheap_exp_gap_vs_policy.png"
    fig.savefig(out, dpi=220)
    print(out)

    # Figure 2: selected compute and error.
    fig, ax1 = plt.subplots(figsize=(12, 7))

    computes = [r["best_compute_mean"] for r in rows]
    computes_err = [r["best_compute_std"] for r in rows]
    errors = [r["best_error_mean"] for r in rows]
    errors_err = [r["best_error_std"] for r in rows]

    ax1.errorbar(x, computes, yerr=computes_err, marker="o", linewidth=2, capsize=5, label="chosen compute")
    ax1.set_ylabel("mean compute cost")
    ax1.set_ylim(0.8, 4.3)

    ax2 = ax1.twinx()
    ax2.errorbar(x, errors, yerr=errors_err, marker="s", linestyle="--", linewidth=2, capsize=5, label="chosen error")
    ax2.set_ylabel("mean prediction error")

    ax1.set_xticks(x)
    ax1.set_xticklabels([short_label(n) for n in names])
    ax1.set_title("Chosen policy changes as the cheap/expensive gap changes")
    ax1.grid(True, alpha=0.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center")

    fig.tight_layout()
    out = FIG_DIR / "cheap_exp_gap_compute_error.png"
    fig.savefig(out, dpi=220)
    print(out)

    # Figure 3: selected fraction.
    fig, ax = plt.subplots(figsize=(12, 7))

    selected = [r["selected_mean"] for r in rows]
    selected_err = [r["selected_std"] for r in rows]

    ax.errorbar(x, selected, yerr=selected_err, marker="o", linewidth=2, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(n) for n in names])
    ax.set_ylabel("expensive predictor activation fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Adaptive execution is only selected in the intermediate regime")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out = FIG_DIR / "cheap_exp_gap_selected_fraction.png"
    fig.savefig(out, dpi=220)
    print(out)


if __name__ == "__main__":
    main()

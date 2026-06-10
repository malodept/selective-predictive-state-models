from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib.pyplot as plt


TABLE_PATH = Path("reports/tables/gain_router_summary.md")
FIG_DIR = Path("reports/figures/gain_router")
OUT_TABLE = Path("reports/tables/gain_router_clean_summary.md")


CORE_METHODS = [
    "cheap-only",
    "all-expensive",
    "learned reliability",
    "action norm",
    "hybrid rel/action alpha=0.50",
    "learned gain small, theoretical threshold",
    "oracle gain upper bound",
]

LABELS = {
    "cheap-only": "cheap-only",
    "all-expensive": "all-expensive",
    "learned reliability": "reliability",
    "action norm": "action norm",
    "hybrid rel/action alpha=0.50": "hybrid",
    "learned gain small, theoretical threshold": "learned gain",
    "oracle gain upper bound": "oracle upper bound",
}

OFFSETS = {
    "cheap-only": (8, 6),
    "all-expensive": (8, 6),
    "learned reliability": (8, 12),
    "action norm": (8, -24),
    "hybrid rel/action alpha=0.50": (-85, 10),
    "learned gain small, theoretical threshold": (8, -10),
    "oracle gain upper bound": (8, 6),
}


def parse_pm(cell: str) -> tuple[float, float]:
    cell = cell.strip().replace("`", "")
    if "±" in cell:
        left, right = cell.split("±", 1)
        return float(left.strip()), float(right.strip())
    return float(cell), 0.0


def parse_table(path: Path) -> dict[str, dict[str, float]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if line.startswith("| ---") or line.startswith("| method"):
            continue

        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 5:
            continue

        method, error, compute, selected, utility = cells
        if method == "method":
            continue

        e, es = parse_pm(error)
        c, cs = parse_pm(compute)
        s, ss = parse_pm(selected)
        u, us = parse_pm(utility)

        rows[method] = {
            "error": e,
            "error_std": es,
            "compute": c,
            "compute_std": cs,
            "selected": s,
            "selected_std": ss,
            "utility": u,
            "utility_std": us,
        }
    return rows


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)

    rows = parse_table(TABLE_PATH)
    missing = [m for m in CORE_METHODS if m not in rows]
    if missing:
        raise RuntimeError(f"Missing methods in {TABLE_PATH}: {missing}")

    core = [(m, rows[m]) for m in CORE_METHODS]

    # ------------------------------------------------------------------
    # Figure 1: clean error--compute trade-off
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    for method, r in core:
        ax.errorbar(
            r["compute"],
            r["error"],
            xerr=r["compute_std"],
            yerr=r["error_std"],
            fmt="o",
            capsize=4,
            markersize=7,
        )
        ax.annotate(
            LABELS[method],
            (r["compute"], r["error"]),
            xytext=OFFSETS[method],
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_title("Gain router and routing baselines: error--compute trade-off")
    ax.set_xlabel("mean compute cost")
    ax.set_ylabel("mean prediction error")
    ax.grid(True, alpha=0.25)

    ax.text(
        0.02,
        0.04,
        "Lower-left is better: lower prediction error and lower compute.",
        transform=ax.transAxes,
        fontsize=9,
        alpha=0.8,
    )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "gain_router_clean_error_compute.png", dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Figure 2: utility improvement over cheap-only
    # ------------------------------------------------------------------
    cheap_u = rows["cheap-only"]["utility"]
    cheap_us = rows["cheap-only"]["utility_std"]

    bar_methods = [
        "all-expensive",
        "random same fraction",
        "learned reliability",
        "action norm",
        "hybrid rel/action alpha=0.25",
        "hybrid rel/action alpha=0.50",
        "hybrid rel/action alpha=0.75",
        "learned gain small, theoretical threshold",
        "learned gain small, calibrated threshold",
        "learned gain state-action, theoretical threshold",
        "learned gain state-action, calibrated threshold",
        "oracle gain upper bound",
    ]
    bar_methods = [m for m in bar_methods if m in rows]

    gains = [rows[m]["utility"] - cheap_u for m in bar_methods]
    gain_stds = [
        math.sqrt(rows[m]["utility_std"] ** 2 + cheap_us**2)
        for m in bar_methods
    ]

    short_names = {
        "all-expensive": "all-exp.",
        "random same fraction": "random",
        "learned reliability": "reliability",
        "action norm": "action norm",
        "hybrid rel/action alpha=0.25": "hybrid .25",
        "hybrid rel/action alpha=0.50": "hybrid .50",
        "hybrid rel/action alpha=0.75": "hybrid .75",
        "learned gain small, theoretical threshold": "gain small",
        "learned gain small, calibrated threshold": "gain small cal.",
        "learned gain state-action, theoretical threshold": "gain s+a",
        "learned gain state-action, calibrated threshold": "gain s+a cal.",
        "oracle gain upper bound": "oracle",
    }

    fig, ax = plt.subplots(figsize=(10, 5.8))
    y = list(range(len(bar_methods)))

    ax.barh(y, gains)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([short_names[m] for m in bar_methods])
    ax.set_xlabel("mean utility improvement over cheap-only")
    ax.set_title("Deployment utility gain at lambda=0.04")
    ax.grid(True, axis="x", alpha=0.25)

    for yi, gain in zip(y, gains):
        ax.text(
            gain + (0.003 if gain >= 0 else -0.003),
            yi,
            f"{gain:+.3f}",
            va="center",
            ha="left" if gain >= 0 else "right",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(FIG_DIR / "gain_router_clean_delta_utility.png", dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Compact markdown table
    # ------------------------------------------------------------------
    lines = [
        "# Clean gain-router diagnostic, DINOv2 best-validation, lambda=0.04",
        "",
        "| method | error | compute | selected | utility | delta utility vs cheap-only |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for method, r in core:
        du = r["utility"] - cheap_u
        lines.append(
            "| "
            + " | ".join(
                [
                    LABELS[method],
                    f"{r['error']:.4f} ± {r['error_std']:.4f}",
                    f"{r['compute']:.4f} ± {r['compute_std']:.4f}",
                    f"{r['selected']:.4f} ± {r['selected_std']:.4f}",
                    f"{r['utility']:.4f} ± {r['utility_std']:.4f}",
                    f"{du:+.4f}",
                ]
            )
            + " |"
        )

    OUT_TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(FIG_DIR / "gain_router_clean_error_compute.png")
    print(FIG_DIR / "gain_router_clean_delta_utility.png")
    print(OUT_TABLE)


if __name__ == "__main__":
    main()

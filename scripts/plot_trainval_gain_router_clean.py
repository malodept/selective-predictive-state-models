from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT = Path("reports/figures/gain_router")
OUT.mkdir(parents=True, exist_ok=True)

# Results from reports/tables/trainval_gain_router_summary.md
rows = [
    {
        "method": "cheap-only",
        "short": "cheap-only",
        "error": 1.2568,
        "error_std": 0.0036,
        "compute": 1.0000,
        "compute_std": 0.0000,
        "delta": 0.0000,
        "delta_std": 0.0000,
    },
    {
        "method": "all-expensive",
        "short": "all-exp.",
        "error": 1.1409,
        "error_std": 0.0007,
        "compute": 4.0000,
        "compute_std": 0.0000,
        "delta": -0.0042,
        "delta_std": 0.0030,
    },
    {
        "method": "reliability threshold",
        "short": "reliability",
        "error": 1.1411,
        "error_std": 0.0006,
        "compute": 3.9915,
        "compute_std": 0.0114,
        "delta": -0.0040,
        "delta_std": 0.0027,
    },
    {
        "method": "action norm threshold",
        "short": "action norm",
        "error": 1.1437,
        "error_std": 0.0008,
        "compute": 3.8329,
        "compute_std": 0.0792,
        "delta": -0.0003,
        "delta_std": 0.0016,
    },
    {
        "method": "hybrid threshold",
        "short": "hybrid",
        "error": 1.1455,
        "error_std": 0.0049,
        "compute": 3.8065,
        "compute_std": 0.2208,
        "delta": -0.0010,
        "delta_std": 0.0006,
    },
    {
        "method": "gain classifier",
        "short": "gain clf.",
        "error": 1.1528,
        "error_std": 0.0041,
        "compute": 3.1464,
        "compute_std": 0.1326,
        "delta": 0.0181,
        "delta_std": 0.0009,
    },
    {
        "method": "gain regressor",
        "short": "gain reg.",
        "error": 1.1613,
        "error_std": 0.0047,
        "compute": 2.9988,
        "compute_std": 0.2015,
        "delta": 0.0155,
        "delta_std": 0.0012,
    },
    {
        "method": "oracle upper bound",
        "short": "oracle",
        "error": 1.0823,
        "error_std": 0.0070,
        "compute": 2.6965,
        "compute_std": 0.0409,
        "delta": 0.1067,
        "delta_std": 0.0093,
    },
]

def get(short):
    return next(r for r in rows if r["short"] == short)

# ---------------------------------------------------------------------
# Figure 1: clean error-compute trade-off
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=220)

plot_order = [
    "cheap-only",
    "gain reg.",
    "gain clf.",
    "oracle",
    "action norm",
    "hybrid",
    "reliability",
    "all-exp.",
]

offsets = {
    "cheap-only": (8, 6),
    "gain reg.": (8, 8),
    "gain clf.": (8, 4),
    "oracle": (8, 3),
    "action norm": (-75, 11),
    "hybrid": (8, 10),
    "reliability": (-75, -8),
    "all-exp.": (-75, -10),
}

for short in plot_order:
    r = get(short)
    ax.errorbar(
        r["compute"],
        r["error"],
        xerr=r["compute_std"],
        yerr=r["error_std"],
        fmt="o",
        capsize=4,
        markersize=6,
        linewidth=1.5,
    )
    ax.annotate(
        short,
        (r["compute"], r["error"]),
        xytext=offsets.get(short, (8, 6)),
        textcoords="offset points",
        fontsize=9,
    )

ax.set_title("Train-to-validation gain router: error--compute trade-off")
ax.set_xlabel("mean compute cost")
ax.set_ylabel("mean prediction error")
ax.text(
    0.02,
    0.035,
    "Lower-left is better.",
    transform=ax.transAxes,
    fontsize=9,
    alpha=0.75,
)
ax.grid(True, alpha=0.25)
ax.set_xlim(0.85, 4.15)
ax.set_ylim(1.065, 1.265)
fig.tight_layout()
fig.savefig(OUT / "trainval_gain_router_error_compute_clean.png", bbox_inches="tight")
plt.close(fig)

# ---------------------------------------------------------------------
# Figure 2: clean utility gain bar plot
# ---------------------------------------------------------------------
bar_order = [
    "oracle",
    "gain clf.",
    "gain reg.",
    "hybrid",
    "action norm",
    "reliability",
    "all-exp.",
    "cheap-only",
]

labels = [get(s)["short"] for s in bar_order]
gains = np.array([get(s)["delta"] for s in bar_order], dtype=float)
gain_stds = np.array([get(s)["delta_std"] for s in bar_order], dtype=float)
y = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=220)
ax.barh(y, gains, xerr=gain_stds, capsize=4)
ax.axvline(0.0, linestyle="--", linewidth=1)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_title("Train-to-validation gain router at lambda=0.04")
ax.set_xlabel("mean utility improvement over cheap-only")
ax.grid(True, axis="x", alpha=0.25)

for yi, g, lab in zip(y, gains, labels):
    # Avoid clutter around the y-axis: only annotate the important positive gains
    # and the cheap-only reference.
    if lab == "cheap-only":
        ax.text(0.002, yi, "+0.000", va="center", ha="left", fontsize=9)
    elif g > 0.004:
        ax.text(g + 0.003, yi, f"{g:+.3f}", va="center", ha="left", fontsize=9)

ax.set_xlim(-0.012, 0.120)
fig.tight_layout()
fig.savefig(OUT / "trainval_gain_router_delta_utility_clean.png", bbox_inches="tight")
plt.close(fig)

print(OUT / "trainval_gain_router_error_compute_clean.png")
print(OUT / "trainval_gain_router_delta_utility_clean.png")

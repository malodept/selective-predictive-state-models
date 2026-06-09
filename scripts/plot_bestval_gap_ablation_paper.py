from __future__ import annotations

import json
from pathlib import Path
import statistics as stats

import matplotlib.pyplot as plt

RUNS = [
    ("cheap5\nexp80", "cheap5_exp80"),
    ("cheap10\nexp80", "cheap10_exp80"),
    ("cheap20\nexp80", "cheap20_exp80"),
    ("cheap10\nexp40", "cheap10_exp40"),
    ("cheap10\nexp120", "cheap10_exp120"),
]
SEEDS = [0, 1, 2]
OUT = Path("reports/figures/bestval_gap_ablation")
OUT.mkdir(parents=True, exist_ok=True)


def load(slug):
    return [
        json.loads(Path(f"outputs/bestval_dinov2_{slug}_seed{seed}/metrics.json").read_text())
        for seed in SEEDS
    ]


def mean_std(xs):
    return stats.mean(xs), stats.stdev(xs) if len(xs) > 1 else 0.0


def vals(ms):
    return [m for m, _ in ms]


def errs(ms):
    return [s for _, s in ms]


labels = []
cheap_error = []
expensive_error = []
gap = []
best_compute = []
selected = []

for label, slug in RUNS:
    ms = load(slug)
    labels.append(label)

    cheap = [m["diagnostics"]["cheap_error"] for m in ms]
    exp = [m["diagnostics"]["expensive_error"] for m in ms]
    gp = [m["diagnostics"]["gap"] for m in ms]
    comp = [m["best_compute"] for m in ms]
    sel = [m["best_selected"] for m in ms]

    cheap_error.append(mean_std(cheap))
    expensive_error.append(mean_std(exp))
    gap.append(mean_std(gp))
    best_compute.append(mean_std(comp))
    selected.append(mean_std(sel))

x = list(range(len(labels)))

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
ax = axes[0, 0]
ax.errorbar(x, vals(cheap_error), yerr=errs(cheap_error), marker="o", capsize=4, label="cheap predictor")
ax.errorbar(x, vals(expensive_error), yerr=errs(expensive_error), marker="s", capsize=4, label="expensive predictor")
ax.set_title("Best-validation stabilizes the expensive predictor")
ax.set_ylabel("prediction error")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(True, alpha=0.25)
ax.legend()

ax = axes[0, 1]
ax.errorbar(x, vals(gap), yerr=errs(gap), marker="o", capsize=4)
ax.axhline(0.0, linestyle="--", linewidth=1)
ax.set_title("Cheap--expensive prediction gap")
ax.set_ylabel("cheap error - expensive error")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(True, alpha=0.25)

ax = axes[1, 0]
ax.errorbar(x, vals(best_compute), yerr=errs(best_compute), marker="o", capsize=4)
ax.set_title("Selected compute cost")
ax.set_ylabel("mean compute cost")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(True, alpha=0.25)

ax = axes[1, 1]
ax.errorbar(x, vals(selected), yerr=errs(selected), marker="o", capsize=4)
ax.set_title("Expensive predictor activation")
ax.set_ylabel("selected fraction")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(True, alpha=0.25)

fig.suptitle("Best-validation selective refinement ablation", fontsize=18)
fig.tight_layout()
path = OUT / "bestval_gap_ablation_summary.png"
fig.savefig(path, dpi=220)
print(path)

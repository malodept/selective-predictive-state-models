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
    ms = []
    for seed in SEEDS:
        path = Path(f"outputs/bestval_dinov2_{slug}_seed{seed}/metrics.json")
        ms.append(json.loads(path.read_text()))
    return ms


def mean_std(xs):
    return stats.mean(xs), stats.stdev(xs) if len(xs) > 1 else 0.0


labels = []
cheap_error = []
expensive_error = []
gap = []
best_error = []
best_compute = []
selected = []

for label, slug in RUNS:
    ms = load(slug)
    labels.append(label)

    ce = [m["diagnostics"]["cheap_error"] for m in ms]
    ee = [m["diagnostics"]["expensive_error"] for m in ms]
    gp = [m["diagnostics"]["gap"] for m in ms]
    be = [m["best_error"] for m in ms]
    bc = [m["best_compute"] for m in ms]
    sel = [m["best_selected"] for m in ms]

    cheap_error.append(mean_std(ce))
    expensive_error.append(mean_std(ee))
    gap.append(mean_std(gp))
    best_error.append(mean_std(be))
    best_compute.append(mean_std(bc))
    selected.append(mean_std(sel))


x = list(range(len(labels)))


def vals(ms):
    return [m for m, _ in ms]


def errs(ms):
    return [s for _, s in ms]


plt.figure(figsize=(12, 6))
plt.errorbar(x, vals(cheap_error), yerr=errs(cheap_error), marker="o", capsize=4, label="cheap")
plt.errorbar(x, vals(expensive_error), yerr=errs(expensive_error), marker="s", capsize=4, label="expensive")
plt.xticks(x, labels)
plt.ylabel("mean prediction error")
plt.title("Best-validation removes expensive-predictor overtraining artifacts")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "bestval_cheap_vs_expensive_error.png", dpi=200)
plt.close()

plt.figure(figsize=(12, 6))
plt.errorbar(x, vals(best_compute), yerr=errs(best_compute), marker="o", capsize=4, label="chosen compute")
plt.xticks(x, labels)
plt.ylabel("mean compute cost")
plt.title("Chosen compute after best-validation checkpoint selection")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "bestval_chosen_compute.png", dpi=200)
plt.close()

plt.figure(figsize=(12, 6))
plt.errorbar(x, vals(selected), yerr=errs(selected), marker="o", capsize=4, label="expensive activation")
plt.xticks(x, labels)
plt.ylabel("selected fraction")
plt.title("Adaptive selector activation after best-validation checkpoint selection")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "bestval_selected_fraction.png", dpi=200)
plt.close()

plt.figure(figsize=(12, 6))
plt.errorbar(x, vals(gap), yerr=errs(gap), marker="o", capsize=4)
plt.axhline(0.0, linestyle="--")
plt.xticks(x, labels)
plt.ylabel("cheap error - expensive error")
plt.title("Positive gap means the expensive predictor genuinely improves prediction")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(OUT / "bestval_gap.png", dpi=200)
plt.close()

print(OUT)
for p in sorted(OUT.glob("*.png")):
    print(p)

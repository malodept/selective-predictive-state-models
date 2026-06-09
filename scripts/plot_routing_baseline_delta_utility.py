from __future__ import annotations

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CSV_PATH = Path("reports/tables/routing_baselines_summary.csv")
FIG_DIR = Path("reports/figures/routing_baselines")
FIG_DIR.mkdir(parents=True, exist_ok=True)

rows = []
with CSV_PATH.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

cheap = next(r for r in rows if r["method"] == "cheap-only")
cheap_u = float(cheap["utility_mean"])

methods = []
deltas = []
stds = []

for r in rows:
    method = r["method"]
    if method == "oracle gain upper bound":
        continue
    methods.append(method)
    deltas.append(float(r["utility_mean"]) - cheap_u)
    stds.append(float(r["utility_std"]))

x = np.arange(len(methods))

plt.figure(figsize=(12, 5))
plt.axhline(0.0, linestyle="--", linewidth=1)
plt.bar(x, deltas, yerr=stds, capsize=4)
plt.xticks(x, methods, rotation=25, ha="right")
plt.ylabel("utility improvement over cheap-only")
plt.title("Routing baselines: adaptive routing improves deployment utility")
plt.grid(axis="y", alpha=0.25)
plt.tight_layout()

out = FIG_DIR / "routing_baselines_delta_utility.png"
plt.savefig(out, dpi=240)
print(out)

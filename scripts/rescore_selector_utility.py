from __future__ import annotations

import json
from pathlib import Path

RUN = Path("outputs/tartanair_resnet18_tiny/metrics.json")

with RUN.open("r", encoding="utf-8") as f:
    metrics = json.load(f)

rows = metrics["selector_rows"]

for lam in [0.0, 0.01, 0.025, 0.04, 0.05, 0.075, 0.1, 0.25]:
    rescored = []
    for r in rows:
        u = -r["mean_error"] - lam * r["mean_compute"]
        rescored.append((u, r["policy"], r["mean_error"], r["mean_compute"], r["selected_fraction"]))

    best = max(rescored, key=lambda x: x[0])
    print(f"\nlambda={lam}")
    print(f"best_policy={best[1]}")
    print(f"utility={best[0]:.4f} error={best[2]:.4f} compute={best[3]:.4f} selected={best[4]:.3f}")

    print("top policies:")
    for u, policy, err, comp, selected in sorted(rescored, reverse=True)[:4]:
        print(f"  {policy:16s} utility={u:.4f} error={err:.4f} compute={comp:.4f} selected={selected:.3f}")

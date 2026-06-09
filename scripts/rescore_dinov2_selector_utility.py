from __future__ import annotations

import json
from pathlib import Path

RUNS = {
    "dinov2_full_heuristic": Path("outputs/tartanair_dinov2_full/metrics.json"),
    "dinov2_error_reliability": Path("outputs/tartanair_dinov2_error_reliability/metrics.json"),
}

LAMBDAS = [0.0, 0.02, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30]

for name, path in RUNS.items():
    print(f"\n=== {name} ===")
    with path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    rows = metrics["selector_rows"]

    for lam in LAMBDAS:
        rescored = []
        for r in rows:
            u = -r["mean_error"] - lam * r["mean_compute"]
            rescored.append((u, r["policy"], r["mean_error"], r["mean_compute"], r["selected_fraction"]))

        best = max(rescored, key=lambda x: x[0])
        print(
            f"lambda={lam:.3f} | best={best[1]:16s} "
            f"utility={best[0]:.4f} error={best[2]:.4f} "
            f"compute={best[3]:.4f} selected={best[4]:.3f}"
        )

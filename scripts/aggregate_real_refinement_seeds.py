from __future__ import annotations

import json
from pathlib import Path
import numpy as np


RUN_DIRS = [
    Path("outputs/real_refinement_dinov2_cheap10_exp80_seed0"),
    Path("outputs/real_refinement_dinov2_cheap10_exp80_seed1"),
    Path("outputs/real_refinement_dinov2_cheap10_exp80_seed2"),
]


def load(path: Path) -> dict:
    with (path / "real_refinement_metrics.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ms = [load(p) for p in RUN_DIRS]

    rows = []
    for seed, m in enumerate(ms):
        ev = m["eval"]
        best = ev["best"]
        rows.append({
            "seed": seed,
            "cheap_error": ev["cheap_error_mean"],
            "expensive_error": ev["expensive_error_mean"],
            "best_policy": best["policy"],
            "best_error": best["mean_error"],
            "best_compute": best["mean_compute"],
            "best_selected": best["selected_fraction"],
            "best_utility": best["utility"],
            "cheap_utility": ev["rows"][0]["utility"],
            "all_expensive_utility": ev["rows"][1]["utility"],
        })

    out = Path("reports/tables/real_refinement_dinov2_multiseed.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    headers = list(rows[0].keys())

    with out.open("w", encoding="utf-8") as f:
        f.write("# DINOv2 real selective refinement multi-seed\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

        for row in rows:
            vals = []
            for h in headers:
                v = row[h]
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            f.write("| " + " | ".join(vals) + " |\n")

        f.write("\n## Summary\n\n")

        numeric = [
            "cheap_error",
            "expensive_error",
            "best_error",
            "best_compute",
            "best_selected",
            "best_utility",
            "cheap_utility",
            "all_expensive_utility",
        ]

        f.write("| metric | mean | std |\n")
        f.write("| --- | ---: | ---: |\n")
        for key in numeric:
            vals = np.asarray([float(r[key]) for r in rows])
            f.write(f"| {key} | {vals.mean():.4f} | {vals.std(ddof=0):.4f} |\n")

    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()

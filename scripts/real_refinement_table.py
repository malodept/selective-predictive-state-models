from __future__ import annotations

import json
from pathlib import Path

RUNS = {
    "DINOv2 cheap10/exp80": Path("outputs/real_refinement_dinov2_cheap10_exp80/real_refinement_metrics.json"),
    "ResNet18 cheap10/exp80": Path("outputs/real_refinement_resnet18_cheap10_exp80/real_refinement_metrics.json"),
}

def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def main() -> None:
    out = Path("reports/tables/real_refinement_table.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for name, path in RUNS.items():
        m = load(path)
        ev = m["eval"]
        best = ev["best"]

        rows.append({
            "run": name,
            "cheap_error": ev["cheap_error_mean"],
            "expensive_error": ev["expensive_error_mean"],
            "best_policy": best["policy"],
            "best_error": best["mean_error"],
            "best_compute": best["mean_compute"],
            "best_selected": best["selected_fraction"],
            "best_utility": best["utility"],
        })

    headers = list(rows[0].keys())

    with out.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

        for row in rows:
            vals = []
            for h in headers:
                v = row[h]
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
            f.write("| " + " | ".join(vals) + " |\n")

    print(out)
    print(out.read_text())

if __name__ == "__main__":
    main()

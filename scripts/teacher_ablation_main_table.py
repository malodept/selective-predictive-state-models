from __future__ import annotations

import json
from pathlib import Path

RUNS = {
    "ResNet18 heuristic": Path("outputs/tartanair_resnet18_full/metrics.json"),
    "ResNet18 error-rel": Path("outputs/tartanair_resnet18_error_reliability/metrics.json"),
    "DINOv2 heuristic": Path("outputs/tartanair_dinov2_full/metrics.json"),
    "DINOv2 error-rel": Path("outputs/tartanair_dinov2_error_reliability/metrics.json"),
}

def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def best_policy(m: dict) -> dict:
    return max(m["selector_rows"], key=lambda r: r["utility"])

def main() -> None:
    rows = []

    for name, path in RUNS.items():
        m = load(path)
        b = best_policy(m)
        r = m["retrieval"]
        s = m["surprise"]

        rows.append({
            "run": name,
            "R@1": r["R@1"],
            "R@5": r["R@5"],
            "R@10": r["R@10"],
            "expected_learned_AUROC": s["expected_learned_auroc"],
            "expected_residual_AUROC": s["expected_residual_auroc"],
            "observed_residual_AUROC": s["observed_residual_auroc"],
            "best_policy": b["policy"],
            "best_error": b["mean_error"],
            "best_compute": b["mean_compute"],
            "best_selected": b["selected_fraction"],
            "best_utility": b["utility"],
        })

    out = Path("reports/tables/teacher_ablation_main_table.md")
    out.parent.mkdir(parents=True, exist_ok=True)

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

from __future__ import annotations

import json
from pathlib import Path


RUNS = {
    "v0.7_full_heuristic": Path("outputs/tartanair_resnet18_full/metrics.json"),
    "v0.8_error_reliability": Path("outputs/tartanair_resnet18_error_reliability/metrics.json"),
}


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    rows = []

    for name, path in RUNS.items():
        m = load(path)
        s = m["surprise"]
        r = m["retrieval"]

        best = max(
            m["selector_rows"],
            key=lambda x: x["utility"],
        )

        rows.append(
            {
                "run": name,
                "R@1": r["R@1"],
                "R@5": r["R@5"],
                "R@10": r["R@10"],
                "expected_learned_AUROC": s["expected_learned_auroc"],
                "expected_residual_AUROC": s["expected_residual_auroc"],
                "observed_residual_AUROC": s["observed_residual_auroc"],
                "best_policy": best["policy"],
                "best_error": best["mean_error"],
                "best_compute": best["mean_compute"],
                "best_selected": best["selected_fraction"],
                "best_utility": best["utility"],
            }
        )

    out = Path("reports/tables/tartanair_v07_v08_comparison.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    headers = list(rows[0].keys())

    with out.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")

        for row in rows:
            values = []
            for h in headers:
                v = row[h]
                if isinstance(v, float):
                    values.append(f"{v:.4f}")
                else:
                    values.append(str(v))
            f.write("| " + " | ".join(values) + " |\n")

    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()

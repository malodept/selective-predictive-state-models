from __future__ import annotations

import json
from pathlib import Path
import numpy as np

RUN_DIRS = [
    Path("outputs/tartanair_dinov2_error_reliability_seed0"),
    Path("outputs/tartanair_dinov2_error_reliability_seed1"),
    Path("outputs/tartanair_dinov2_error_reliability_seed2"),
]

METRICS = [
    ("R@1", lambda m: m["retrieval"]["R@1"]),
    ("R@5", lambda m: m["retrieval"]["R@5"]),
    ("R@10", lambda m: m["retrieval"]["R@10"]),
    ("expected_learned_auroc", lambda m: m["surprise"]["expected_learned_auroc"]),
    ("expected_residual_auroc", lambda m: m["surprise"]["expected_residual_auroc"]),
    ("observed_residual_auroc", lambda m: m["surprise"]["observed_residual_auroc"]),
    ("observed_learned_auroc", lambda m: m["surprise"]["observed_learned_auroc"]),
]

def load(path: Path) -> dict:
    with (path / "metrics.json").open("r", encoding="utf-8") as f:
        return json.load(f)

def best_policy(m: dict) -> dict:
    return max(m["selector_rows"], key=lambda r: r["utility"])

def main() -> None:
    ms = [load(p) for p in RUN_DIRS]

    out = Path("reports/tables/dinov2_errorrel_multiseed_summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        f.write("# DINOv2 error-supervised reliability multi-seed summary\n\n")
        f.write("| metric | mean | std | values |\n")
        f.write("| --- | ---: | ---: | --- |\n")

        for name, fn in METRICS:
            vals = np.asarray([float(fn(m)) for m in ms])
            f.write(
                f"| {name} | {vals.mean():.4f} | {vals.std(ddof=0):.4f} | "
                + ", ".join(f"{v:.4f}" for v in vals)
                + " |\n"
            )

        f.write("\n## Best selector policy per seed, fixed lambda from config\n\n")
        f.write("| seed | policy | error | compute | selected | utility |\n")
        f.write("| --- | --- | ---: | ---: | ---: | ---: |\n")

        for seed, m in enumerate(ms):
            b = best_policy(m)
            f.write(
                f"| {seed} | {b['policy']} | {b['mean_error']:.4f} | "
                f"{b['mean_compute']:.4f} | {b['selected_fraction']:.4f} | "
                f"{b['utility']:.4f} |\n"
            )

    print(out)
    print(out.read_text())

if __name__ == "__main__":
    main()

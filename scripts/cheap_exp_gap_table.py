from __future__ import annotations

import json
from pathlib import Path


RUNS = {
    "cheap5/exp80": Path("outputs/real_refinement_dinov2_cheap5_exp80_seed0"),
    "cheap10/exp80": Path("outputs/real_refinement_dinov2_cheap10_exp80_seed0"),
    "cheap20/exp80": Path("outputs/real_refinement_dinov2_cheap20_exp80_seed0"),
    "cheap10/exp40": Path("outputs/real_refinement_dinov2_cheap10_exp40_seed0"),
    "cheap10/exp120": Path("outputs/real_refinement_dinov2_cheap10_exp120_seed0"),
}


def load_metrics(path: Path) -> dict:
    metrics_path = path / "real_refinement_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    return json.loads(metrics_path.read_text())


def main() -> None:
    out = Path("reports/tables/cheap_exp_gap_ablation_seed0.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Cheap/expensive gap ablation, DINOv2, seed 0\n")
    lines.append("| run | cheap_error | expensive_error | gap | best_policy | best_error | best_compute | selected | utility |")
    lines.append("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")

    for name, run_dir in RUNS.items():
        m = load_metrics(run_dir)
        ev = m["eval"]
        best = ev["best"]

        cheap_error = float(ev["cheap_error_mean"])
        expensive_error = float(ev["expensive_error_mean"])
        gap = cheap_error - expensive_error

        lines.append(
            f"| {name} | "
            f"{cheap_error:.4f} | "
            f"{expensive_error:.4f} | "
            f"{gap:.4f} | "
            f"{best['policy']} | "
            f"{best['mean_error']:.4f} | "
            f"{best['mean_compute']:.4f} | "
            f"{best['selected_fraction']:.4f} | "
            f"{best['utility']:.4f} |"
        )

    out.write_text("\n".join(lines) + "\n")
    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()

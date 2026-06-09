from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev


CONFIGS = {
    "cheap5/exp80": "cheap5_exp80",
    "cheap10/exp80": "cheap10_exp80",
    "cheap20/exp80": "cheap20_exp80",
    "cheap10/exp40": "cheap10_exp40",
    "cheap10/exp120": "cheap10_exp120",
}

SEEDS = [0, 1, 2]


def load_metrics(cfg: str, seed: int) -> dict:
    path = Path(f"outputs/real_refinement_dinov2_{cfg}_seed{seed}/real_refinement_metrics.json")
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def mean_std(xs: list[float]) -> tuple[float, float]:
    return mean(xs), pstdev(xs)


def policy_mode(policies: list[str]) -> str:
    return max(set(policies), key=policies.count)


def main() -> None:
    out = Path("reports/tables/cheap_exp_gap_ablation_multiseed.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Cheap/expensive gap ablation, DINOv2, multi-seed\n")
    lines.append("| run | cheap error | expensive error | gap | best policy mode | best error | best compute | selected | utility | policies |")
    lines.append("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |")

    csv_lines = []
    csv_lines.append(
        "run,cheap_error_mean,cheap_error_std,expensive_error_mean,expensive_error_std,"
        "gap_mean,gap_std,best_policy_mode,best_error_mean,best_error_std,"
        "best_compute_mean,best_compute_std,selected_mean,selected_std,"
        "utility_mean,utility_std,policies"
    )

    for display_name, cfg in CONFIGS.items():
        cheap_errors = []
        expensive_errors = []
        gaps = []
        best_errors = []
        best_computes = []
        selected = []
        utilities = []
        policies = []

        for seed in SEEDS:
            m = load_metrics(cfg, seed)
            ev = m["eval"]
            best = ev["best"]

            cheap = float(ev["cheap_error_mean"])
            exp = float(ev["expensive_error_mean"])

            cheap_errors.append(cheap)
            expensive_errors.append(exp)
            gaps.append(cheap - exp)

            best_errors.append(float(best["mean_error"]))
            best_computes.append(float(best["mean_compute"]))
            selected.append(float(best["selected_fraction"]))
            utilities.append(float(best["utility"]))
            policies.append(str(best["policy"]))

        cheap_m, cheap_s = mean_std(cheap_errors)
        exp_m, exp_s = mean_std(expensive_errors)
        gap_m, gap_s = mean_std(gaps)
        best_err_m, best_err_s = mean_std(best_errors)
        compute_m, compute_s = mean_std(best_computes)
        selected_m, selected_s = mean_std(selected)
        utility_m, utility_s = mean_std(utilities)
        pol_mode = policy_mode(policies)
        pol_str = ", ".join(policies)

        lines.append(
            f"| {display_name} | "
            f"{cheap_m:.4f} ± {cheap_s:.4f} | "
            f"{exp_m:.4f} ± {exp_s:.4f} | "
            f"{gap_m:.4f} ± {gap_s:.4f} | "
            f"{pol_mode} | "
            f"{best_err_m:.4f} ± {best_err_s:.4f} | "
            f"{compute_m:.4f} ± {compute_s:.4f} | "
            f"{selected_m:.4f} ± {selected_s:.4f} | "
            f"{utility_m:.4f} ± {utility_s:.4f} | "
            f"{pol_str} |"
        )

        csv_lines.append(
            f"{display_name},{cheap_m:.6f},{cheap_s:.6f},{exp_m:.6f},{exp_s:.6f},"
            f"{gap_m:.6f},{gap_s:.6f},{pol_mode},{best_err_m:.6f},{best_err_s:.6f},"
            f"{compute_m:.6f},{compute_s:.6f},{selected_m:.6f},{selected_s:.6f},"
            f"{utility_m:.6f},{utility_s:.6f},\"{pol_str}\""
        )

    out.write_text("\n".join(lines) + "\n")
    Path("reports/tables/cheap_exp_gap_ablation_multiseed.csv").write_text("\n".join(csv_lines) + "\n")

    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()

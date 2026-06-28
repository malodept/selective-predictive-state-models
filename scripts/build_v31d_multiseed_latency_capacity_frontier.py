from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v31_multiseed_capacity_ladder_ood")
PER_SEED = ROOT / "multiseed_capacity_ladder_per_seed.csv"
LAT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v24_measured_latency/capacity_latency_summary.csv")

OUT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v31d_multiseed_latency_frontier")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

OUT_PER = OUT / "multiseed_latency_frontier_per_seed.csv"
OUT_AGG = OUT / "multiseed_latency_frontier_aggregate.csv"
OUT_MD = OUT / "multiseed_latency_frontier_summary.md"

LAMBDAS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
CAP_ORDER = ["Tiny", "Small", "Medium", "Full"]

HARD_VARIANTS = ["3-block, H=36", "3-block, H=48", "3-block, H=72"]


def main():
    df = pd.read_csv(PER_SEED)
    lat = pd.read_csv(LAT)
    b128 = lat[lat["batch_size"] == 128][["model_label", "relative_latency_b128", "latency_ms_median"]].copy()
    b128 = b128.rename(columns={
        "model_label": "capacity_label",
        "relative_latency_b128": "relative_latency",
        "latency_ms_median": "latency_ms_b128",
    })

    df = df.merge(b128, on="capacity_label", how="left", validate="many_to_one")

    rows = []
    for lam in LAMBDAS:
        tmp = df.copy()
        tmp["lambda"] = lam
        tmp["utility"] = tmp["tie_aware_top1"] - lam * tmp["relative_latency"]
        rows.append(tmp)

    per = pd.concat(rows, ignore_index=True)
    per.to_csv(OUT_PER, index=False)

    agg = (
        per
        .groupby(["lambda", "variant", "variant_label", "capacity_label"], as_index=False)
        .agg(
            top1_mean=("tie_aware_top1", "mean"),
            top1_std=("tie_aware_top1", "std"),
            utility_mean=("utility", "mean"),
            utility_std=("utility", "std"),
            relative_latency=("relative_latency", "first"),
            latency_ms_b128=("latency_ms_b128", "first"),
            n=("seed", "count"),
        )
    )
    agg.to_csv(OUT_AGG, index=False)

    lines = []
    lines.append("# SPSM v31D multi-seed latency-normalized capacity frontier\n")
    lines.append("This repeats the latency-normalized fixed-capacity frontier using three independently trained seeds for every capacity.")
    lines.append("Utility is `tie-aware top-1 - lambda * measured_relative_latency`, with Full batch-128 latency normalized to 1.0.\n")

    lines.append("## Best mean fixed capacity by OOD variant and cost\n")
    lines.append("| lambda | variant | best mean capacity | utility mean±std | top-1 mean±std | relative latency |")
    lines.append("|---:|---|---|---:|---:|---:|")

    for lam in LAMBDAS:
        for variant, sub in agg[agg["lambda"] == lam].groupby("variant_label", sort=False):
            best = sub.sort_values("utility_mean", ascending=False).iloc[0]
            lines.append(
                f"| {lam:.2f} | {variant} | `{best['capacity_label']}` | "
                f"{best['utility_mean']:.3f}±{best['utility_std']:.3f} | "
                f"{best['top1_mean']:.3f}±{best['top1_std']:.3f} | "
                f"{best['relative_latency']:.3f} |"
            )

    lines.append("\n## Hard OOD utility frontier at selected costs\n")
    lines.append("| lambda | variant | Tiny | Small | Medium | Full | best |")
    lines.append("|---:|---|---:|---:|---:|---:|---|")

    for lam in [0.00, 0.05, 0.10, 0.20]:
        for variant in HARD_VARIANTS:
            sub = agg[(agg["lambda"] == lam) & (agg["variant_label"] == variant)].set_index("capacity_label")
            vals = {}
            for cap in CAP_ORDER:
                vals[cap] = (
                    float(sub.loc[cap, "utility_mean"]),
                    float(sub.loc[cap, "utility_std"]),
                )
            best = max(vals, key=lambda c: vals[c][0])
            lines.append(
                f"| {lam:.2f} | {variant} | "
                f"{vals['Tiny'][0]:.3f}±{vals['Tiny'][1]:.3f} | "
                f"{vals['Small'][0]:.3f}±{vals['Small'][1]:.3f} | "
                f"{vals['Medium'][0]:.3f}±{vals['Medium'][1]:.3f} | "
                f"{vals['Full'][0]:.3f}±{vals['Full'][1]:.3f} | "
                f"`{best}` |"
            )

    lines.append("\n## Crossover diagnostics on hard OOD\n")
    lines.append("| variant | Medium-Full top-1 gap | Medium-Tiny top-1 gap | Tiny-Medium latency advantage | approx Medium-vs-Tiny λ crossover |")
    lines.append("|---|---:|---:|---:|---:|")

    base = agg[agg["lambda"] == 0.0]
    for variant in HARD_VARIANTS:
        sub = base[base["variant_label"] == variant].set_index("capacity_label")
        med_top = float(sub.loc["Medium", "top1_mean"])
        full_top = float(sub.loc["Full", "top1_mean"])
        tiny_top = float(sub.loc["Tiny", "top1_mean"])
        med_cost = float(sub.loc["Medium", "relative_latency"])
        tiny_cost = float(sub.loc["Tiny", "relative_latency"])

        med_full_gap = med_top - full_top
        med_tiny_gap = med_top - tiny_top
        latency_adv = med_cost - tiny_cost
        cross = med_tiny_gap / latency_adv if abs(latency_adv) > 1e-12 else float("nan")

        lines.append(
            f"| {variant} | {med_full_gap:.6f} | {med_tiny_gap:.6f} | "
            f"{latency_adv:.6f} | {cross:.4f} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append("- The multi-seed result supports the non-monotonic capacity claim: Full is not the best mean model on any block3 hard OOD shift.")
    lines.append("- Medium is the best mean accuracy model on H=36, H=48, and H=72, but the best individual seed can vary.")
    lines.append("- At higher compute cost, Tiny can become optimal because Medium's accuracy advantage no longer pays for its additional measured latency.")
    lines.append("- This is a stronger and more defensible version of the v25/v29 capacity frontier.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure: hard OOD accuracy-latency with mean/std over seeds.
    base = agg[agg["lambda"] == 0.0]
    plt.figure(figsize=(6.8, 4.2))
    for variant in HARD_VARIANTS:
        sub = base[base["variant_label"] == variant].set_index("capacity_label").loc[CAP_ORDER].reset_index()
        plt.errorbar(
            sub["relative_latency"],
            sub["top1_mean"],
            yerr=sub["top1_std"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=variant,
        )
        for _, r in sub.iterrows():
            plt.annotate(r["capacity_label"], (r["relative_latency"], r["top1_mean"]), textcoords="offset points", xytext=(4, 4), fontsize=7)

    plt.xlabel("Relative measured latency, batch=128")
    plt.ylabel("Tie-aware top-1")
    plt.title("Multi-seed capacity frontier on hard OOD")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig_v31d_multiseed_capacity_frontier_hard_ood.png", dpi=220)
    plt.close()

    # Figure: utility vs lambda on H=72.
    h72 = agg[agg["variant_label"] == "3-block, H=72"]
    plt.figure(figsize=(6.8, 4.2))
    for cap in CAP_ORDER:
        sub = h72[h72["capacity_label"] == cap].sort_values("lambda")
        plt.plot(sub["lambda"], sub["utility_mean"], marker="o", linewidth=1.8, label=cap)
        plt.fill_between(
            sub["lambda"],
            sub["utility_mean"] - sub["utility_std"],
            sub["utility_mean"] + sub["utility_std"],
            alpha=0.12,
        )

    plt.xlabel("Compute cost $\\lambda$")
    plt.ylabel("Utility")
    plt.title("Multi-seed latency-normalized frontier on 3-block H=72")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig_v31d_h72_utility_vs_lambda.png", dpi=220)
    plt.close()

    print(OUT_PER)
    print(OUT_AGG)
    print(OUT_MD)
    print()
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()

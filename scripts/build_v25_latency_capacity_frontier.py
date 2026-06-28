from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ACC_IN = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v22_capacity_ladder_ood/capacity_ladder_ood_summary.csv")
LAT_IN = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v24_measured_latency/capacity_latency_summary.csv")

OUT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v25_latency_capacity_frontier")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT / "latency_capacity_frontier.csv"
OUT_MD = OUT / "latency_capacity_frontier.md"

LAMBDAS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
MODEL_ORDER = ["Tiny", "Small", "Medium", "Full"]


def main():
    acc = pd.read_csv(ACC_IN)
    lat = pd.read_csv(LAT_IN)

    b128 = lat[lat["batch_size"] == 128].copy()
    b128 = b128[["model_label", "latency_ms_median", "relative_latency_b128", "params", "checkpoint_mb"]]

    df = acc.merge(b128, on="model_label", how="left", validate="many_to_one")

    rows = []
    for lam in LAMBDAS:
        for variant, sub in df.groupby("variant_label", sort=False):
            for _, r in sub.iterrows():
                rows.append({
                    "lambda": lam,
                    "variant": variant,
                    "model": r["model_label"],
                    "top1": r["tie_aware_top1"],
                    "latency_ms_b128": r["latency_ms_median"],
                    "relative_latency": r["relative_latency_b128"],
                    "checkpoint_mb": r["checkpoint_mb_y"],
                    "params": r["params"],
                    "utility": r["tie_aware_top1"] - lam * r["relative_latency_b128"],
                    "same_action_diff_state_win": r["same_action_diff_state_win"],
                    "same_state_diff_action_win": r["same_state_diff_action_win"],
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# SPSM v25 latency-normalized capacity frontier\n")
    lines.append("This rebuilds the v23 fixed-capacity frontier using measured batch-128 forward latency instead of checkpoint size.")
    lines.append("The objective is `utility = top1 - lambda * relative_latency`, where Full latency is normalized to 1.0.\n")

    lines.append("## Measured latency-normalized best fixed capacity\n")
    lines.append("| lambda | variant | best model | utility | top-1 | relative latency | latency ms @128 |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")

    for lam in LAMBDAS:
        for variant, sub in out[out["lambda"] == lam].groupby("variant", sort=False):
            best = sub.sort_values("utility", ascending=False).iloc[0]
            lines.append(
                f"| {lam:.2f} | {variant} | `{best['model']}` | "
                f"{best['utility']:.6f} | {best['top1']:.6f} | "
                f"{best['relative_latency']:.6f} | {best['latency_ms_b128']:.4f} |"
            )

    lines.append("\n## Accuracy-latency table\n")
    lines.append("| variant | model | top-1 | latency ms @128 | relative latency | params |")
    lines.append("|---|---|---:|---:|---:|---:|")

    base = out[out["lambda"] == 0.0].copy()
    for variant, sub in base.groupby("variant", sort=False):
        sub = sub.set_index("model").loc[MODEL_ORDER].reset_index()
        for _, r in sub.iterrows():
            lines.append(
                f"| {variant} | {r['model']} | {r['top1']:.6f} | "
                f"{r['latency_ms_b128']:.4f} | {r['relative_latency']:.6f} | {int(r['params'])} |"
            )

    lines.append("\n## Key crossovers\n")
    lines.append("| variant | comparison | lambda crossover | interpretation |")
    lines.append("|---|---|---:|---|")

    for variant, sub in base.groupby("variant", sort=False):
        piv = sub.set_index("model")
        pairs = [("Full", "Tiny"), ("Medium", "Tiny"), ("Medium", "Small"), ("Small", "Tiny")]
        for a, b in pairs:
            acc_a = float(piv.loc[a, "top1"])
            acc_b = float(piv.loc[b, "top1"])
            cost_a = float(piv.loc[a, "relative_latency"])
            cost_b = float(piv.loc[b, "relative_latency"])
            denom = cost_a - cost_b
            if abs(denom) < 1e-12:
                cross = float("nan")
                interp = "same measured cost"
            else:
                cross = (acc_a - acc_b) / denom
                if cross > 0:
                    interp = f"{a} preferred below crossover, {b} above"
                else:
                    interp = "higher-cost model is not justified by accuracy"
            lines.append(f"| {variant} | {a} vs {b} | {cross:.4f} | {interp} |")

    lines.append("\n## Interpretation\n")
    lines.append("- Measured latency makes Tiny/Small less cheap than checkpoint size suggested, so v23 overstated the cost advantage of Tiny.")
    lines.append("- Capacity remains non-monotonic: Medium is the best accuracy model on the hardest 3-block H=72 shift, while Full is not consistently best.")
    lines.append("- For low compute cost, Medium remains valuable on hard long-horizon OOD; for high cost, Tiny becomes optimal.")
    lines.append("- This supports a multi-capacity value-of-computation formulation: choosing among Tiny, Small, Medium, and Full is more faithful than binary Small-vs-Full routing.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Main hard-OOD frontier figure.
    hard = base[base["variant"].isin(["3-block, H=36", "3-block, H=48", "3-block, H=72"])].copy()
    plt.figure(figsize=(6.8, 4.2))
    for variant, sub in hard.groupby("variant", sort=False):
        sub = sub.set_index("model").loc[MODEL_ORDER].reset_index()
        plt.plot(sub["relative_latency"], sub["top1"], marker="o", linewidth=1.8, label=variant)
        for _, r in sub.iterrows():
            plt.annotate(r["model"], (r["relative_latency"], r["top1"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    plt.xlabel("Relative measured latency, batch=128")
    plt.ylabel("Tie-aware top-1")
    plt.title("Measured accuracy-latency frontier on hard OOD")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig_v25_measured_latency_frontier_hard_ood.png", dpi=220)
    plt.close()

    # Best model map by lambda.
    best_rows = []
    for lam in LAMBDAS:
        for variant, sub in out[out["lambda"] == lam].groupby("variant", sort=False):
            best = sub.sort_values("utility", ascending=False).iloc[0]
            best_rows.append({"lambda": lam, "variant": variant, "best": best["model"]})
    best_df = pd.DataFrame(best_rows)
    best_df.to_csv(OUT / "best_model_by_lambda.csv", index=False)

    print(OUT_MD)
    print(OUT_CSV)
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v22_capacity_ladder_ood/capacity_ladder_ood_summary.csv")

OUT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v23_capacity_cost_frontier")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT / "capacity_cost_frontier.csv"
OUT_MD = OUT / "capacity_cost_frontier.md"

LAMBDAS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]

MODEL_ORDER = ["Tiny", "Small", "Medium", "Full"]


def main():
    df = pd.read_csv(IN)

    # Use checkpoint size as a first cost proxy.
    # Later, replace this with measured inference latency.
    full_mb = float(df[df["model_label"] == "Full"]["checkpoint_mb"].iloc[0])
    df["cost_proxy"] = df["checkpoint_mb"] / full_mb

    rows = []
    for lam in LAMBDAS:
        for variant, sub in df.groupby("variant_label", sort=False):
            for _, r in sub.iterrows():
                rows.append({
                    "lambda": lam,
                    "variant": variant,
                    "model": r["model_label"],
                    "top1": r["tie_aware_top1"],
                    "checkpoint_mb": r["checkpoint_mb"],
                    "cost_proxy": r["cost_proxy"],
                    "utility": r["tie_aware_top1"] - lam * r["cost_proxy"],
                    "same_action_diff_state_win": r["same_action_diff_state_win"],
                    "same_state_diff_action_win": r["same_state_diff_action_win"],
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# SPSM v23 capacity cost frontier\n")
    lines.append("This is an aggregate fixed-model frontier: choose one capacity for all examples in a variant.")
    lines.append("Cost is currently approximated by checkpoint size relative to Full. This is a proxy; measured latency should replace it later.\n")

    lines.append("## Best fixed capacity by OOD variant and cost\n")
    lines.append("| lambda | variant | best model | utility | top-1 | relative cost |")
    lines.append("|---:|---|---|---:|---:|---:|")

    for lam in LAMBDAS:
        for variant, sub in out[out["lambda"] == lam].groupby("variant", sort=False):
            best = sub.sort_values("utility", ascending=False).iloc[0]
            lines.append(
                f"| {lam:.2f} | {variant} | `{best['model']}` | "
                f"{best['utility']:.6f} | {best['top1']:.6f} | {best['cost_proxy']:.6f} |"
            )

    lines.append("\n## Accuracy-cost table\n")
    lines.append("| variant | model | top-1 | checkpoint MB | relative cost | same-action/diff-state | same-state/diff-action |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")

    base = out[out["lambda"] == 0.0].copy()
    for variant, sub in base.groupby("variant", sort=False):
        sub = sub.set_index("model").loc[MODEL_ORDER].reset_index()
        for _, r in sub.iterrows():
            lines.append(
                f"| {variant} | {r['model']} | {r['top1']:.6f} | "
                f"{r['checkpoint_mb']:.2f} | {r['cost_proxy']:.6f} | "
                f"{r['same_action_diff_state_win']:.6f} | {r['same_state_diff_action_win']:.6f} |"
            )

    lines.append("\n## Interpretation\n")
    lines.append("- Capacity is not monotonic: Medium is best on the hardest 3-block H=72 shift, while Full is not consistently best.")
    lines.append("- Because Full has much higher relative cost, any positive compute penalty makes Medium/Tiny/Small more attractive unless Full has a large accuracy advantage.")
    lines.append("- This supports replacing binary small-to-full routing with a multi-capacity value-of-computation problem.")
    lines.append("- The next rigorous step is to measure actual inference latency and then build per-instance routing across capacities.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Figure: accuracy vs cost per variant.
    for variant, sub in base.groupby("variant", sort=False):
        sub = sub.set_index("model").loc[MODEL_ORDER].reset_index()

        plt.figure(figsize=(5.7, 3.8))
        plt.plot(sub["cost_proxy"], sub["top1"], marker="o", linewidth=1.8)
        for _, r in sub.iterrows():
            plt.annotate(r["model"], (r["cost_proxy"], r["top1"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
        plt.xlabel("Relative cost proxy")
        plt.ylabel("Tie-aware top-1")
        plt.title(f"Capacity frontier: {variant}")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()

        safe = (
            variant
            .replace(",", "")
            .replace("=", "")
            .replace(" ", "_")
            .replace("-", "")
            .lower()
        )
        plt.savefig(FIG / f"fig_v23_capacity_frontier_{safe}.png", dpi=220)
        plt.close()

    # Figure: hard variants together.
    hard = base[base["variant"].isin(["3-block, H=36", "3-block, H=48", "3-block, H=72"])]
    plt.figure(figsize=(6.7, 4.2))
    for variant, sub in hard.groupby("variant", sort=False):
        sub = sub.set_index("model").loc[MODEL_ORDER].reset_index()
        plt.plot(sub["cost_proxy"], sub["top1"], marker="o", linewidth=1.8, label=variant)
    plt.xlabel("Relative cost proxy")
    plt.ylabel("Tie-aware top-1")
    plt.title("Capacity frontier on hard OOD shifts")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "fig_v23_capacity_frontier_hard_ood.png", dpi=220)
    plt.close()

    print(OUT_MD)
    print(OUT_CSV)
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()

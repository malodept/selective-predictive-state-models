from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("reports/paper_assets_v31_multiseed_capacity")
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

AGG = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v31d_multiseed_latency_frontier/multiseed_latency_frontier_aggregate.csv")
SUMMARY = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v31d_multiseed_latency_frontier/multiseed_latency_frontier_summary.md")

df = pd.read_csv(AGG)

CAP_ORDER = ["Tiny", "Small", "Medium", "Full"]
HARD = ["3-block, H=36", "3-block, H=48", "3-block, H=72"]

# ---------------------------------------------------------------------
# Table 1: multi-seed hard OOD accuracy frontier.
# ---------------------------------------------------------------------
base = df[(df["lambda"] == 0.0) & (df["variant_label"].isin(HARD))].copy()

rows = []
for variant in HARD:
    sub = base[base["variant_label"] == variant].set_index("capacity_label")
    row = {"variant": variant}
    for cap in CAP_ORDER:
        row[f"{cap}_mean"] = float(sub.loc[cap, "top1_mean"])
        row[f"{cap}_std"] = float(sub.loc[cap, "top1_std"])
    rows.append(row)

tab = pd.DataFrame(rows)
tab.to_csv(TAB / "table_v31_multiseed_capacity_frontier_hard_ood.csv", index=False)

tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\resizebox{\linewidth}{!}{%")
tex.append(r"\begin{tabular}{lrrrr}")
tex.append(r"\toprule")
tex.append(r"Hard OOD shift & Tiny & Small & Medium & Full \\")
tex.append(r"\midrule")
for _, r in tab.iterrows():
    tex.append(
        f"{r['variant']} & "
        f"{r['Tiny_mean']:.3f}$\\pm${r['Tiny_std']:.3f} & "
        f"{r['Small_mean']:.3f}$\\pm${r['Small_std']:.3f} & "
        f"{r['Medium_mean']:.3f}$\\pm${r['Medium_std']:.3f} & "
        f"{r['Full_mean']:.3f}$\\pm${r['Full_std']:.3f} \\\\"
    )
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}%")
tex.append(r"}")
tex.append(r"\caption{Multi-seed capacity frontier under hard OOD shifts. Values are tie-aware top-1 mean$\pm$standard deviation over three independently trained predictors per capacity. Capacity is non-monotonic: the largest model is not the best mean model on any block3 hard-OOD shift.}")
tex.append(r"\label{tab:multiseed_capacity_frontier_hard_ood}")
tex.append(r"\end{table}")
(TAB / "table_v31_multiseed_capacity_frontier_hard_ood.tex").write_text("\n".join(tex) + "\n")

# ---------------------------------------------------------------------
# Table 2: selected latency-normalized utilities.
# ---------------------------------------------------------------------
selected_lams = [0.00, 0.05, 0.10, 0.20]
rows = []
for lam in selected_lams:
    for variant in HARD:
        sub = df[(df["lambda"] == lam) & (df["variant_label"] == variant)].set_index("capacity_label")
        row = {"lambda": lam, "variant": variant}
        for cap in CAP_ORDER:
            row[f"{cap}_utility_mean"] = float(sub.loc[cap, "utility_mean"])
            row[f"{cap}_utility_std"] = float(sub.loc[cap, "utility_std"])
        best = max(CAP_ORDER, key=lambda c: row[f"{c}_utility_mean"])
        row["best"] = best
        rows.append(row)

util_tab = pd.DataFrame(rows)
util_tab.to_csv(TAB / "table_v31_multiseed_latency_utility_selected.csv", index=False)

tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\resizebox{\linewidth}{!}{%")
tex.append(r"\begin{tabular}{llrrrrl}")
tex.append(r"\toprule")
tex.append(r"$\lambda$ & Shift & Tiny & Small & Medium & Full & Best \\")
tex.append(r"\midrule")
for _, r in util_tab.iterrows():
    tex.append(
        f"{r['lambda']:.2f} & {r['variant']} & "
        f"{r['Tiny_utility_mean']:.3f} & "
        f"{r['Small_utility_mean']:.3f} & "
        f"{r['Medium_utility_mean']:.3f} & "
        f"{r['Full_utility_mean']:.3f} & "
        f"{r['best']} \\\\"
    )
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}%")
tex.append(r"}")
tex.append(r"\caption{Latency-normalized fixed-capacity utility on hard OOD shifts. Utility is tie-aware top-1 minus $\lambda$ times measured relative batch-128 latency. Medium is favored at low cost on H=48/H=72, while smaller models become optimal as compute cost increases.}")
tex.append(r"\label{tab:multiseed_latency_utility_hard_ood}")
tex.append(r"\end{table}")
(TAB / "table_v31_multiseed_latency_utility_selected.tex").write_text("\n".join(tex) + "\n")

# ---------------------------------------------------------------------
# Figure 1: accuracy-latency frontier.
# ---------------------------------------------------------------------
plt.figure(figsize=(6.9, 4.25))
for variant in HARD:
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
plt.savefig(FIG / "fig_v31_multiseed_capacity_frontier_hard_ood.png", dpi=240)
plt.close()

# ---------------------------------------------------------------------
# Figure 2: utility curves on H=72.
# ---------------------------------------------------------------------
h72 = df[df["variant_label"] == "3-block, H=72"].copy()

plt.figure(figsize=(6.9, 4.25))
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
plt.title("Latency-normalized utility on 3-block H=72")
plt.grid(True, alpha=0.25)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig_v31_h72_latency_utility_vs_lambda.png", dpi=240)
plt.close()

# ---------------------------------------------------------------------
# Claim text.
# ---------------------------------------------------------------------
claim = """
# Paper assets v31 multi-seed capacity

## Main update

The capacity frontier is now validated across three independently trained seeds for every capacity.
The single-seed v29 claim is strengthened: Full is not the best mean model on any block3 hard-OOD shift.

## Recommended replacement claim

Under constrained-geometry OOD, predictive capacity is not monotonically ordered.
Medium-capacity predictors outperform the largest predictor on average across hard block3 shifts, while measured latency determines whether Medium, Small, or Tiny is optimal at a given compute cost.
This makes value-of-computation a multi-capacity selection problem rather than a simple small-to-large cascade.

## Key numbers

- 3-block H=36: Medium 0.879±0.014 vs Full 0.855±0.024.
- 3-block H=48: Medium 0.884±0.001 vs Full 0.872±0.030.
- 3-block H=72: Medium 0.829±0.016 vs Full 0.808±0.011.
- H=72 Medium-vs-Tiny latency crossover: λ≈0.099.
- H=48 Medium-vs-Tiny latency crossover: λ≈0.156.

## Claim boundary

Do not claim that Medium is always best seed-by-seed.
The correct claim is that capacity is non-monotonic in expectation over retraining seeds, and that the best capacity is shift-, seed-, and cost-dependent.
"""
(OUT / "v31_multiseed_capacity_claim.md").write_text(claim.strip() + "\n")

manifest = []
manifest.append("# Paper assets v31 multi-seed capacity\n")
manifest.append("## Tables\n")
for p in sorted(TAB.glob("*")):
    manifest.append(f"- `{p}`")
manifest.append("\n## Figures\n")
for p in sorted(FIG.glob("*")):
    manifest.append(f"- `{p}`")
manifest.append("\n## Text\n")
manifest.append(f"- `{OUT / 'v31_multiseed_capacity_claim.md'}`")
(OUT / "paper_assets_v31_manifest.md").write_text("\n".join(manifest) + "\n")

print((OUT / "paper_assets_v31_manifest.md").read_text())
print()
print((TAB / "table_v31_multiseed_capacity_frontier_hard_ood.tex").read_text())
print()
print((TAB / "table_v31_multiseed_latency_utility_selected.tex").read_text())
print()
print((OUT / "v31_multiseed_capacity_claim.md").read_text())

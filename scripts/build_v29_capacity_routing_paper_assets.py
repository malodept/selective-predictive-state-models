from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("reports/paper_assets_v29_capacity_routing")
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

V25 = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v25_latency_capacity_frontier/latency_capacity_frontier.csv")
V28 = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v28_multicapacity_router_bootstrap/multicapacity_router_bootstrap_summary.csv")

frontier = pd.read_csv(V25)
boot = pd.read_csv(V28)

# ---------------------------------------------------------------------
# Table 1: measured latency frontier on hard OOD at lambda = 0.
# ---------------------------------------------------------------------
hard_variants = ["3-block, H=36", "3-block, H=48", "3-block, H=72"]
models = ["Tiny", "Small", "Medium", "Full"]

base = frontier[(frontier["lambda"] == 0.0) & (frontier["variant"].isin(hard_variants))].copy()

rows = []
for variant in hard_variants:
    sub = base[base["variant"] == variant].set_index("model")
    row = {"variant": variant}
    for m in models:
        row[f"{m}_top1"] = float(sub.loc[m, "top1"])
        row[f"{m}_lat"] = float(sub.loc[m, "relative_latency"])
    rows.append(row)

table_frontier = pd.DataFrame(rows)
table_frontier.to_csv(TAB / "table_v29_hard_ood_capacity_frontier.csv", index=False)

tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\resizebox{\linewidth}{!}{%")
tex.append(r"\begin{tabular}{lrrrr}")
tex.append(r"\toprule")
tex.append(r"Hard OOD shift & Tiny & Small & Medium & Full \\")
tex.append(r"\midrule")
for _, r in table_frontier.iterrows():
    tex.append(
        f"{r['variant']} & "
        f"{r['Tiny_top1']:.3f} & {r['Small_top1']:.3f} & "
        f"{r['Medium_top1']:.3f} & {r['Full_top1']:.3f} \\\\"
    )
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}%")
tex.append(r"}")
tex.append(r"\caption{Capacity is non-monotonic under hard OOD shifts. Values are tie-aware top-1; all models are evaluated with the same intervention protocol. Measured batch-128 relative latencies are Tiny=0.186, Small=0.201, Medium=0.383, Full=1.000.}")
tex.append(r"\label{tab:capacity_frontier_hard_ood}")
tex.append(r"\end{table}")
(TAB / "table_v29_hard_ood_capacity_frontier.tex").write_text("\n".join(tex) + "\n")

# ---------------------------------------------------------------------
# Table 2: bootstrap learned routing summary.
# ---------------------------------------------------------------------
learned = boot[
    boot["method"].isin([
        "ridge_expected_utility",
        "rf_expected_utility",
        "logreg_oracle_label",
        "rf_oracle_label",
    ])
].copy()

best_rows = []
for lam in [0.00, 0.05, 0.10, 0.20, 0.30]:
    for variant in hard_variants:
        sub = learned[(learned["lambda"] == lam) & (learned["variant_label"] == variant)]
        best = sub.sort_values("utility", ascending=False).iloc[0]
        best_rows.append({
            "lambda": lam,
            "variant": variant,
            "method": best["method"],
            "utility": best["utility"],
            "delta": best["delta_vs_fixed"],
            "delta_lo": best["delta_ci95_low"],
            "delta_hi": best["delta_ci95_high"],
            "stable": bool(best["stable_positive_vs_fixed"]),
            "oracle_gap": best["oracle_gap"],
        })

best_df = pd.DataFrame(best_rows)
best_df.to_csv(TAB / "table_v29_learned_router_bootstrap_best.csv", index=False)

tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\resizebox{\linewidth}{!}{%")
tex.append(r"\begin{tabular}{llrrr}")
tex.append(r"\toprule")
tex.append(r"$\lambda$ & Shift & Best learned router & $\Delta$ vs fixed & 95\% CI \\")
tex.append(r"\midrule")
for _, r in best_df.iterrows():
    if r["lambda"] in [0.05, 0.10, 0.20]:
        method = str(r["method"]).replace("_", "-")
        stable = r"\textbf{yes}" if r["stable"] else "no"
        tex.append(
            f"{r['lambda']:.2f} & {r['variant']} & {method} & "
            f"{r['delta']:.3f} & [{r['delta_lo']:.3f}, {r['delta_hi']:.3f}] ({stable}) \\\\"
        )
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}%")
tex.append(r"}")
tex.append(r"\caption{Bootstrap validation of learned multi-capacity routing. The table reports the best learned router on each held-out hard OOD shift and its paired utility gain over the best fixed capacity. Stable gains appear on the H=48 shift, while H=72 retains substantial oracle headroom.}")
tex.append(r"\label{tab:learned_multicapacity_bootstrap}")
tex.append(r"\end{table}")
(TAB / "table_v29_learned_router_bootstrap_best.tex").write_text("\n".join(tex) + "\n")

# ---------------------------------------------------------------------
# Figure 1: measured latency frontier.
# ---------------------------------------------------------------------
plt.figure(figsize=(6.8, 4.2))
for variant in hard_variants:
    sub = base[base["variant"] == variant].set_index("model").loc[models].reset_index()
    plt.plot(sub["relative_latency"], sub["top1"], marker="o", linewidth=1.8, label=variant)
    for _, r in sub.iterrows():
        plt.annotate(r["model"], (r["relative_latency"], r["top1"]), textcoords="offset points", xytext=(4, 4), fontsize=7)

plt.xlabel("Relative measured latency, batch=128")
plt.ylabel("Tie-aware top-1")
plt.title("Non-monotonic capacity frontier on hard OOD")
plt.grid(True, alpha=0.25)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig_v29_capacity_frontier_hard_ood.png", dpi=220)
plt.close()

# ---------------------------------------------------------------------
# Figure 2: learned routing gain over fixed.
# ---------------------------------------------------------------------
plt.figure(figsize=(6.8, 4.2))
for variant in hard_variants:
    sub = best_df[best_df["variant"] == variant].sort_values("lambda")
    plt.plot(sub["lambda"], sub["delta"], marker="o", linewidth=1.8, label=variant)
    plt.fill_between(sub["lambda"], sub["delta_lo"], sub["delta_hi"], alpha=0.12)

plt.axhline(0.0, linewidth=1.0)
plt.xlabel("Compute cost $\\lambda$")
plt.ylabel("Utility gain over best fixed")
plt.title("Learned multi-capacity routing gain")
plt.grid(True, alpha=0.25)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig_v29_learned_router_gain_bootstrap.png", dpi=220)
plt.close()

# ---------------------------------------------------------------------
# Manifest.
# ---------------------------------------------------------------------
md = []
md.append("# Paper assets v29 capacity routing\n")
md.append("## Scientific update\n")
md.append("The capacity ladder shows non-monotonic OOD reliability: Medium beats Full on the hardest H=72 shift, while Tiny is often the best low-cost choice.")
md.append("The learned multi-capacity router has stable positive gains on H=48, but not on H=36 or H=72. H=72 remains an oracle-headroom problem.\n")

md.append("## Tables\n")
for p in sorted(TAB.glob("*")):
    md.append(f"- `{p}`")

md.append("\n## Figures\n")
for p in sorted(FIG.glob("*")):
    md.append(f"- `{p}`")

md.append("\n## Recommended claim\n")
md.append("SPSM reveals non-monotonic capacity under controlled OOD interventions and shows that learned multi-capacity routing can provide statistically stable gains on specific hard shifts, while harder regimes expose remaining oracle headroom and the need for stronger routing signals.")

(OUT / "paper_assets_v29_manifest.md").write_text("\n".join(md) + "\n")

print((OUT / "paper_assets_v29_manifest.md").read_text())
print()
print((TAB / "table_v29_hard_ood_capacity_frontier.tex").read_text())
print()
print((TAB / "table_v29_learned_router_bootstrap_best.tex").read_text())

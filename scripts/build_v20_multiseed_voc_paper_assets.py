from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v19_multiseed_locked_voc/locked_voc_multiseed_summary.csv")

OUT = Path("reports/paper_assets_v20_multiseed_voc")
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

h72 = df[
    (df["heldout"] == "block3_h72_seed14")
    & (df["lambda"].isin([0.02, 0.05, 0.10, 0.20, 0.30]))
].copy()

method_order = [
    "cheap",
    "full",
    "confidence",
    "context_cls",
    "knn_context",
    "rh_context",
    "knn_context_latent",
    "rh_context_latent",
    "oracle",
]

method_labels = {
    "cheap": "Cheap",
    "full": "Full",
    "confidence": "Conf.",
    "context_cls": "Ctx-cls",
    "knn_context": "KNN-Ctx",
    "rh_context": "RH-Ctx",
    "knn_context_latent": "KNN-Ctx+Lat",
    "rh_context_latent": "RH-Ctx+Lat",
    "oracle": "Oracle",
}

# ---------------------------------------------------------------------
# CSV compact table
# ---------------------------------------------------------------------
wide_rows = []
for lam in [0.02, 0.05, 0.10, 0.20, 0.30]:
    sub = h72[h72["lambda"] == lam].set_index("method")
    row = {"lambda": lam}
    for m in method_order:
        row[f"{m}_mean"] = float(sub.loc[m, "utility_mean"])
        row[f"{m}_std"] = float(sub.loc[m, "utility_std"])
        row[f"{m}_route"] = float(sub.loc[m, "route_rate_mean"])
    wide_rows.append(row)

wide = pd.DataFrame(wide_rows)
wide.to_csv(TAB / "table_v20_h72_multiseed_locked_voc.csv", index=False)

# ---------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------
tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\resizebox{\linewidth}{!}{%")
tex.append(r"\begin{tabular}{lrrrrrrrrr}")
tex.append(r"\toprule")
tex.append(r"$\lambda$ & Cheap & Full & Conf. & Ctx-cls & KNN-Ctx & RH-Ctx & KNN-Ctx+Lat & RH-Ctx+Lat & Oracle \\")
tex.append(r"\midrule")

for _, r in wide.iterrows():
    lam = r["lambda"]
    vals = []
    for m in method_order:
        vals.append(f"{r[f'{m}_mean']:.3f}$\\pm${r[f'{m}_std']:.3f}")
    tex.append(f"{lam:.2f} & " + " & ".join(vals) + r" \\")

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}%")
tex.append(r"}")
tex.append(r"\caption{Multi-cheap-seed locked value-of-computation comparison on the hardest OOD shift, 3-block H=72. Values are mean$\pm$standard deviation over three independently trained cheap predictors; the expensive predictor is fixed.}")
tex.append(r"\label{tab:multiseed_voc_h72}")
tex.append(r"\end{table}")

(TAB / "table_v20_h72_multiseed_locked_voc.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

# ---------------------------------------------------------------------
# Figure 1: utility curves with std bands
# ---------------------------------------------------------------------
plot_methods = [
    "cheap",
    "full",
    "confidence",
    "knn_context",
    "rh_context",
    "knn_context_latent",
    "rh_context_latent",
    "oracle",
]

plt.figure(figsize=(7.6, 4.6))
for method in plot_methods:
    sub = h72[h72["method"] == method].sort_values("lambda")
    x = sub["lambda"].to_numpy()
    y = sub["utility_mean"].to_numpy()
    e = sub["utility_std"].fillna(0.0).to_numpy()
    plt.plot(x, y, marker="o", linewidth=1.8, label=method_labels[method])
    plt.fill_between(x, y - e, y + e, alpha=0.12)

plt.xlabel("Compute cost $\\lambda$")
plt.ylabel("Utility")
plt.title("Multi-seed VoC routing on 3-block, H=72")
plt.grid(True, alpha=0.25)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(FIG / "fig_v20_multiseed_utility_h72.png", dpi=220)
plt.close()

# ---------------------------------------------------------------------
# Figure 2: latent advantage over context-only
# ---------------------------------------------------------------------
deltas = []
per_seed = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v19_multiseed_locked_voc/locked_voc_per_cheap_seed.csv")
ps = pd.read_csv(per_seed)
h72_ps = ps[ps["heldout"] == "block3_h72_seed14"]

piv = h72_ps.pivot_table(
    index=["cheap_seed", "lambda"],
    columns="method",
    values="utility",
).reset_index()

for lam in [0.02, 0.05, 0.10, 0.20, 0.30]:
    sub = piv[piv["lambda"] == lam]
    deltas.append({
        "lambda": lam,
        "knn_mean": float((sub["knn_context_latent"] - sub["knn_context"]).mean()),
        "knn_std": float((sub["knn_context_latent"] - sub["knn_context"]).std(ddof=1)),
        "rh_mean": float((sub["rh_context_latent"] - sub["rh_context"]).mean()),
        "rh_std": float((sub["rh_context_latent"] - sub["rh_context"]).std(ddof=1)),
    })

d = pd.DataFrame(deltas)
d.to_csv(TAB / "table_v20_latent_advantage_h72.csv", index=False)

plt.figure(figsize=(6.8, 4.1))
plt.axhline(0.0, linewidth=1.0)
plt.errorbar(d["lambda"], d["knn_mean"], yerr=d["knn_std"], marker="o", linewidth=1.8, capsize=3, label="KNN latent - context")
plt.errorbar(d["lambda"], d["rh_mean"], yerr=d["rh_std"], marker="o", linewidth=1.8, capsize=3, label="RH latent - context")
plt.xlabel("Compute cost $\\lambda$")
plt.ylabel("Utility gain")
plt.title("Latent-state advantage over context-only routing")
plt.grid(True, alpha=0.25)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig_v20_latent_advantage_h72.png", dpi=220)
plt.close()

# ---------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------
md = []
md.append("# Paper assets v20 multi-seed VoC\n")
md.append("## Main scientific update\n")
md.append("The single-seed v15 result becomes a multi-cheap-seed stability result.")
md.append("On 3-block H=72, routing remains useful, but the best router varies across cheap-model seeds.")
md.append("Latent-state features provide a positive mean advantage at several compute costs, especially for KNN at λ=0.05–0.20, but the effect is small relative to cheap-seed variability.\n")

md.append("## Tables\n")
for p in sorted(TAB.glob("*")):
    md.append(f"- `{p}`")

md.append("\n## Figures\n")
for p in sorted(FIG.glob("*")):
    md.append(f"- `{p}`")

md.append("\n## Recommended paper claim\n")
md.append("Use: observable latent-state features give directionally positive routing gains on the hardest long-horizon shift, but multi-seed results show that cheap-model realization is a major factor. The strongest robust contribution is therefore value-of-computation routing under non-uniform cheap/full model ordering, with latent-state features as a promising stabilizing signal.")

(OUT / "paper_assets_v20_manifest.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print((OUT / "paper_assets_v20_manifest.md").read_text())
print()
print((TAB / "table_v20_h72_multiseed_locked_voc.tex").read_text())

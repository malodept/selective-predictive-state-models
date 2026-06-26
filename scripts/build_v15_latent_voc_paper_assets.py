from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

IN = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v14_locked_router_comparison/locked_router_comparison.csv")
OUT = Path("reports/paper_assets_v15_latent_voc")
FIG = OUT / "figures"
TAB = OUT / "tables"

FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

# Clean labels.
df["lambda_label"] = df["lambda"].map(lambda x: f"{x:.2f}")

# Main compact table: the important H=72 cases.
main = df[
    (df["heldout"] == "block3_h72_seed14")
    & (df["lambda"].isin([0.02, 0.05, 0.10, 0.20, 0.30]))
].copy()

cols = [
    "lambda",
    "cheap",
    "full",
    "oracle",
    "conf",
    "context_cls",
    "knn_context",
    "rh_context",
    "knn_context_latent",
    "rh_context_latent",
    "best_fixed",
]
main[cols].to_csv(TAB / "table_v15_h72_locked_router.csv", index=False)

# All locked results for archive.
df.to_csv(TAB / "table_v15_locked_router_all.csv", index=False)

# LaTeX table for direct paper inclusion.
tex_lines = []
tex_lines.append(r"\begin{table}[t]")
tex_lines.append(r"\centering")
tex_lines.append(r"\small")
tex_lines.append(r"\resizebox{\linewidth}{!}{%")
tex_lines.append(r"\begin{tabular}{lrrrrrrrrr}")
tex_lines.append(r"\toprule")
tex_lines.append(r"$\lambda$ & Cheap & Full & Conf. & Ctx-cls & KNN-Ctx & RH-Ctx & KNN-Ctx+Lat & RH-Ctx+Lat & Oracle \\")
tex_lines.append(r"\midrule")
for _, r in main.iterrows():
    tex_lines.append(
        f"{r['lambda']:.2f} & "
        f"{r['cheap']:.3f} & {r['full']:.3f} & {r['conf']:.3f} & {r['context_cls']:.3f} & "
        f"{r['knn_context']:.3f} & {r['rh_context']:.3f} & "
        f"{r['knn_context_latent']:.3f} & {r['rh_context_latent']:.3f} & "
        f"{r['oracle']:.3f} \\\\"
    )
tex_lines.append(r"\bottomrule")
tex_lines.append(r"\end{tabular}%")
tex_lines.append(r"}")
tex_lines.append(r"\caption{Locked value-of-computation comparison on the hardest long-horizon OOD shift, 3-block H=72. Context+latent routers use only observable pre-routing information: cheap-model risk, environment context, and current latent-state statistics.}")
tex_lines.append(r"\label{tab:latent_voc_h72}")
tex_lines.append(r"\end{table}")
(TAB / "table_v15_h72_locked_router.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

# Figure 1: utility vs lambda on H=72.
h72 = df[df["heldout"] == "block3_h72_seed14"].sort_values("lambda")
methods = [
    ("cheap", "Cheap"),
    ("full", "Full"),
    ("conf", "Confidence"),
    ("context_cls", "Context-cls"),
    ("knn_context", "KNN context"),
    ("rh_context", "RH context"),
    ("knn_context_latent", "KNN context+latent"),
    ("rh_context_latent", "RH context+latent"),
    ("oracle", "Oracle"),
]

plt.figure(figsize=(7.2, 4.5))
for col, label in methods:
    plt.plot(h72["lambda"], h72[col], marker="o", linewidth=1.8, label=label)
plt.xlabel("Compute cost $\\lambda$")
plt.ylabel("Utility")
plt.title("Locked VoC routing on 3-block, H=72")
plt.grid(True, alpha=0.25)
plt.legend(fontsize=8, ncol=2)
plt.tight_layout()
plt.savefig(FIG / "fig_v15_locked_utility_h72.png", dpi=220)
plt.close()

# Figure 2: latent improvement deltas by heldout.
delta_cols = [
    ("delta_knn_latent_vs_context", "KNN latent - context"),
    ("delta_rh_latent_vs_context", "RH latent - context"),
    ("delta_best_latent_vs_best_context", "Best latent - best context"),
]

for heldout, label in [
    ("block3_h36_seed13", "3-block, H=36"),
    ("block3_h48_seed10", "3-block, H=48"),
    ("block3_h72_seed14", "3-block, H=72"),
]:
    sub = df[df["heldout"] == heldout].sort_values("lambda")
    plt.figure(figsize=(6.6, 3.9))
    for col, name in delta_cols:
        plt.axhline(0.0, linewidth=1.0)
        plt.plot(sub["lambda"], sub[col], marker="o", linewidth=1.8, label=name)
    plt.xlabel("Compute cost $\\lambda$")
    plt.ylabel("Utility gain")
    plt.title(f"Latent-state improvement over context-only: {label}")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    safe_map = {
        "block3_h36_seed13": "block3_h36",
        "block3_h48_seed10": "block3_h48",
        "block3_h72_seed14": "block3_h72",
    }
    safe = safe_map[heldout]
    plt.savefig(FIG / f"fig_v15_latent_improvement_{safe}.png", dpi=220)
    plt.close()

# Figure 3: compact bar chart at the strongest H=72 case lambda=0.10.
case = df[(df["heldout"] == "block3_h72_seed14") & (df["lambda"] == 0.10)].iloc[0]
bar_methods = [
    ("cheap", "Cheap"),
    ("full", "Full"),
    ("conf", "Conf."),
    ("context_cls", "Ctx-cls"),
    ("knn_context", "KNN-Ctx"),
    ("rh_context", "RH-Ctx"),
    ("knn_context_latent", "KNN-Ctx+Lat"),
    ("rh_context_latent", "RH-Ctx+Lat"),
    ("oracle", "Oracle"),
]
plt.figure(figsize=(8.0, 4.2))
plt.bar([x[1] for x in bar_methods], [case[x[0]] for x in bar_methods])
plt.ylabel("Utility")
plt.title("3-block, H=72 at $\\lambda=0.10$")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(FIG / "fig_v15_h72_lambda010_bar.png", dpi=220)
plt.close()

manifest = []
manifest.append("# Paper assets v15 latent VoC\n")
manifest.append("## Tables\n")
for p in sorted(TAB.glob("*")):
    manifest.append(f"- `{p}`")
manifest.append("\n## Figures\n")
for p in sorted(FIG.glob("*")):
    manifest.append(f"- `{p}`")
(OUT / "paper_assets_v15_manifest.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")

print((OUT / "paper_assets_v15_manifest.md").read_text())

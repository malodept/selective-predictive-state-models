from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SRC = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v33_ood_grid/v33d_full_grid_aggregate.csv")

OUT = Path("reports/paper_assets_v33_ood_grid")
TAB = OUT / "tables"
FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC)

CAP_ORDER = ["Tiny", "Small", "Medium", "Full"]

# ---------------------------------------------------------------------
# Table: compact OOD grid summary.
# ---------------------------------------------------------------------
rows = []
for horizon in [36, 72]:
    for blocked in [1, 2, 3, 4]:
        sub = df[(df["horizon"] == horizon) & (df["blocked"] == blocked)].set_index("model_label")
        row = {
            "horizon": horizon,
            "blocked": blocked,
        }
        for cap in CAP_ORDER:
            row[f"{cap}_mean"] = float(sub.loc[cap, "top1_mean"])
            row[f"{cap}_std"] = float(sub.loc[cap, "top1_std"])
        vals = {cap: row[f"{cap}_mean"] for cap in CAP_ORDER}
        row["best"] = max(vals, key=vals.get)
        row["range"] = max(vals.values()) - min(vals.values())
        row["medium_minus_full"] = row["Medium_mean"] - row["Full_mean"]
        rows.append(row)

tab = pd.DataFrame(rows)
tab.to_csv(TAB / "table_v33_ood_grid_capacity_summary.csv", index=False)

tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\resizebox{\linewidth}{!}{%")
tex.append(r"\begin{tabular}{rrrrrrrl}")
tex.append(r"\toprule")
tex.append(r"Horizon & Blocked & Tiny & Small & Medium & Full & Med.-Full & Best \\")
tex.append(r"\midrule")
for _, r in tab.iterrows():
    tex.append(
        f"{int(r['horizon'])} & {int(r['blocked'])} & "
        f"{r['Tiny_mean']:.3f} & "
        f"{r['Small_mean']:.3f} & "
        f"{r['Medium_mean']:.3f} & "
        f"{r['Full_mean']:.3f} & "
        f"{r['medium_minus_full']:+.3f} & "
        f"{r['best']} \\\\"
    )
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}%")
tex.append(r"}")
tex.append(r"\caption{Systematic OOD grid over blocked directions and horizons. Values are tie-aware top-1 averaged over three layout seeds. Capacity ordering is non-monotonic: Full is best only in the easy blocked=2 regime, Medium dominates several intermediate constrained-geometry regimes, and smaller models are favored in the extreme blocked=4 regime.}")
tex.append(r"\label{tab:ood_grid_capacity_summary}")
tex.append(r"\end{table}")
(TAB / "table_v33_ood_grid_capacity_summary.tex").write_text("\n".join(tex) + "\n")

# ---------------------------------------------------------------------
# Table: decomposition for H=72.
# ---------------------------------------------------------------------
h72 = df[df["horizon"] == 72].copy()
h72.to_csv(TAB / "table_v33_h72_decomposition.csv", index=False)

tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\resizebox{\linewidth}{!}{%")
tex.append(r"\begin{tabular}{rlrrr}")
tex.append(r"\toprule")
tex.append(r"Blocked & Model & Top-1 & Same-action/diff-state & Same-state/diff-action \\")
tex.append(r"\midrule")
for blocked in [1, 2, 3, 4]:
    sub = h72[h72["blocked"] == blocked].set_index("model_label")
    for cap in CAP_ORDER:
        r = sub.loc[cap]
        tex.append(
            f"{blocked} & {cap} & "
            f"{r['top1_mean']:.3f} & "
            f"{r['same_action_mean']:.3f} & "
            f"{r['same_state_mean']:.3f} \\\\"
        )
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}%")
tex.append(r"}")
tex.append(r"\caption{Failure decomposition on the horizon-72 OOD grid. The same-action/different-state term degrades more strongly than the same-state/different-action term, indicating that constrained geometry primarily harms state discrimination under fixed actions.}")
tex.append(r"\label{tab:ood_grid_h72_decomposition}")
tex.append(r"\end{table}")
(TAB / "table_v33_h72_decomposition.tex").write_text("\n".join(tex) + "\n")

# ---------------------------------------------------------------------
# Copy/regenerate figures in paper asset location.
# ---------------------------------------------------------------------
for horizon in [36, 72]:
    plt.figure(figsize=(6.8, 4.2))
    for cap in CAP_ORDER:
        sub = df[(df["horizon"] == horizon) & (df["model_label"] == cap)].sort_values("blocked")
        plt.errorbar(
            sub["blocked"],
            sub["top1_mean"],
            yerr=sub["top1_std"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=cap,
        )
    plt.xlabel("Blocked directions per state")
    plt.ylabel("Tie-aware top-1")
    plt.title(f"Systematic OOD grid, horizon={horizon}")
    plt.xticks([1, 2, 3, 4])
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / f"fig_v33_grid_top1_vs_blocked_h{horizon}.png", dpi=240)
    plt.close()

plt.figure(figsize=(6.8, 4.2))
for cap in CAP_ORDER:
    sub = df[(df["horizon"] == 72) & (df["model_label"] == cap)].sort_values("blocked")
    plt.errorbar(
        sub["blocked"],
        sub["same_action_mean"],
        yerr=sub["same_action_std"],
        marker="o",
        linewidth=1.8,
        capsize=3,
        label=cap,
    )
plt.xlabel("Blocked directions per state")
plt.ylabel("same-action / different-state win")
plt.title("State discrimination under constrained geometry, H=72")
plt.xticks([1, 2, 3, 4])
plt.grid(True, alpha=0.25)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig_v33_h72_same_action_state_discrimination.png", dpi=240)
plt.close()

claim = """
# Paper assets v33 OOD grid

## Main update

The OOD result now uses a systematic mini-grid rather than a small set of hand-picked variants:
blocked directions {1,2,3,4}, horizons {36,72}, and three layout seeds.

## Recommended claim

Controlled OOD geometry induces non-linear reliability regimes.
Capacity is not monotonically ordered across the grid: Full is best mainly in the easy blocked=2 regime, Medium dominates several intermediate constrained-geometry regimes, and smaller capacities become competitive or best in the extreme blocked=4 regime.

## Key numbers

- H=36, blocked=3: Medium 0.927 vs Full 0.869.
- H=72, blocked=1: Medium 0.927 vs Full 0.879.
- H=72, blocked=3: Medium 0.839 vs Full 0.813.
- H=72, blocked=4: Small 0.602 vs Full 0.543.
- blocked=4 gives nearly identical H=36/H=72 behavior, consistent with horizon becoming irrelevant when all moving directions are blocked.

## Claim boundary

Do not claim monotonic degradation with blocked count. The grid shows non-linear geometry regimes, not a simple scalar difficulty axis.
Do not claim Medium is universally best. The correct claim is that capacity ordering is shift-dependent and non-monotonic.
"""
(OUT / "v33_ood_grid_claim.md").write_text(claim.strip() + "\n")

manifest = []
manifest.append("# Paper assets v33 OOD grid\n")
manifest.append("## Tables\n")
for p in sorted(TAB.glob("*")):
    manifest.append(f"- `{p}`")
manifest.append("\n## Figures\n")
for p in sorted(FIG.glob("*")):
    manifest.append(f"- `{p}`")
manifest.append("\n## Text\n")
manifest.append(f"- `{OUT / 'v33_ood_grid_claim.md'}`")
(OUT / "paper_assets_v33_manifest.md").write_text("\n".join(manifest) + "\n")

print((OUT / "paper_assets_v33_manifest.md").read_text())
print()
print((TAB / "table_v33_ood_grid_capacity_summary.tex").read_text())
print()
print((TAB / "table_v33_h72_decomposition.tex").read_text())
print()
print((OUT / "v33_ood_grid_claim.md").read_text())

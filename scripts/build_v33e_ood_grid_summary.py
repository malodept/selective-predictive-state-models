from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v33_ood_grid")
OUT_CSV = ROOT / "v33d_full_grid_per_variant_model.csv"
OUT_AGG = ROOT / "v33d_full_grid_aggregate.csv"
OUT_MD = ROOT / "v33d_full_grid_summary.md"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("tiny_full_seed0", "Tiny"),
    ("small_full_seed0", "Small"),
    ("medium_full_seed0", "Medium"),
    ("full_seed0", "Full"),
]

CAP_ORDER = ["Tiny", "Small", "Medium", "Full"]


def parse_md(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    out = {}

    for line in txt.splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([-+0-9.eE]+)\s*\|", line)
        if m:
            key = m.group(1).strip()
            val = float(m.group(2))
            if key == "biased argmin top-1":
                out["biased_argmin_top1"] = val
            elif key == "strict top-1":
                out["strict_top1"] = val
            elif key == "tie-aware top-1":
                out["tie_aware_top1"] = val
            elif key == "mean tie count at minimum":
                out["mean_tie_count"] = val

        if "same_action_diff_state" in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) >= 7:
                out["same_action_diff_state_win"] = float(parts[1])
                out["same_action_diff_state_margin"] = float(parts[5])
                out["same_action_diff_state_count"] = int(float(parts[6]))

        if "same_state_diff_action" in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) >= 7:
                out["same_state_diff_action_win"] = float(parts[1])
                out["same_state_diff_action_margin"] = float(parts[5])
                out["same_state_diff_action_count"] = int(float(parts[6]))

    return out


rows = []

for blocked in [1, 2, 3, 4]:
    for horizon in [36, 72]:
        for layout_seed in [20, 21, 22]:
            variant = f"block{blocked}_h{horizon}_seed{layout_seed}"
            for model, label in MODELS:
                path = ROOT / variant / f"moving_only_{model}.md"
                if not path.exists():
                    raise FileNotFoundError(path)

                vals = parse_md(path)
                rows.append({
                    "variant": variant,
                    "blocked": blocked,
                    "horizon": horizon,
                    "layout_seed": layout_seed,
                    "model": model,
                    "model_label": label,
                    **vals,
                })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

agg = (
    df.groupby(["blocked", "horizon", "model_label"], as_index=False)
    .agg(
        top1_mean=("tie_aware_top1", "mean"),
        top1_std=("tie_aware_top1", "std"),
        same_action_mean=("same_action_diff_state_win", "mean"),
        same_action_std=("same_action_diff_state_win", "std"),
        same_state_mean=("same_state_diff_action_win", "mean"),
        same_state_std=("same_state_diff_action_win", "std"),
        n=("variant", "count"),
    )
)

agg["model_label"] = pd.Categorical(agg["model_label"], categories=CAP_ORDER, ordered=True)
agg = agg.sort_values(["horizon", "blocked", "model_label"])
agg.to_csv(OUT_AGG, index=False)

lines = []
lines.append("# SPSM v33D full OOD mini-grid summary\n")
lines.append("This summarizes the systematic OOD grid: blocked directions `{1,2,3,4}`, horizons `{36,72}`, layout seeds `{20,21,22}`, and four seed0 capacities.\n")

lines.append("## Mean tie-aware top-1 over layout seeds\n")
lines.append("| horizon | blocked | Tiny | Small | Medium | Full | best mean | range |")
lines.append("|---:|---:|---:|---:|---:|---:|---|---:|")

for horizon in [36, 72]:
    for blocked in [1, 2, 3, 4]:
        sub = agg[(agg["horizon"] == horizon) & (agg["blocked"] == blocked)].set_index("model_label")
        vals = {m: float(sub.loc[m, "top1_mean"]) for m in CAP_ORDER}
        stds = {m: float(sub.loc[m, "top1_std"]) for m in CAP_ORDER}
        best = max(vals, key=vals.get)
        spread = max(vals.values()) - min(vals.values())

        lines.append(
            f"| {horizon} | {blocked} | "
            f"{vals['Tiny']:.3f}±{stds['Tiny']:.3f} | "
            f"{vals['Small']:.3f}±{stds['Small']:.3f} | "
            f"{vals['Medium']:.3f}±{stds['Medium']:.3f} | "
            f"{vals['Full']:.3f}±{stds['Full']:.3f} | "
            f"`{best}` | {spread:.3f} |"
        )

lines.append("\n## State/action decomposition\n")
lines.append("| horizon | blocked | model | same-action/diff-state | same-state/diff-action |")
lines.append("|---:|---:|---|---:|---:|")

for horizon in [36, 72]:
    for blocked in [1, 2, 3, 4]:
        sub = agg[(agg["horizon"] == horizon) & (agg["blocked"] == blocked)].sort_values("model_label")
        for _, r in sub.iterrows():
            lines.append(
                f"| {horizon} | {blocked} | {r['model_label']} | "
                f"{r['same_action_mean']:.3f}±{r['same_action_std']:.3f} | "
                f"{r['same_state_mean']:.3f}±{r['same_state_std']:.3f} |"
            )

lines.append("\n## Best capacity per individual variant\n")
lines.append("| variant | Tiny | Small | Medium | Full | best |")
lines.append("|---|---:|---:|---:|---:|---|")

for variant, sub in df.groupby("variant"):
    s = sub.set_index("model_label")
    vals = {m: float(s.loc[m, "tie_aware_top1"]) for m in CAP_ORDER}
    best = max(vals, key=vals.get)
    lines.append(
        f"| {variant} | {vals['Tiny']:.3f} | {vals['Small']:.3f} | "
        f"{vals['Medium']:.3f} | {vals['Full']:.3f} | `{best}` |"
    )

# Key diagnostics
lines.append("\n## Key diagnostics\n")
for horizon in [36, 72]:
    lines.append(f"\n### Horizon {horizon}\n")
    for blocked in [1, 2, 3, 4]:
        sub = agg[(agg["horizon"] == horizon) & (agg["blocked"] == blocked)].set_index("model_label")
        med_full = float(sub.loc["Medium", "top1_mean"] - sub.loc["Full", "top1_mean"])
        best = sub.sort_values("top1_mean", ascending=False).iloc[0]
        lines.append(
            f"- blocked={blocked}: best={best.name}, Medium-Full gap={med_full:.4f}."
        )

lines.append("\n## Interpretation guide\n")
lines.append("- If performance decreases monotonically with blocked directions, the grid validates constrained geometry as a systematic difficulty axis.")
lines.append("- If Full is not consistently best across blocked/horizon cells, the non-monotonic capacity result generalizes beyond hand-picked OOD shifts.")
lines.append("- If the same-action/diff-state term degrades faster than same-state/diff-action, the main failure remains state discrimination under fixed action.")

OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

# Figure 1: top-1 vs blocked for each horizon/capacity.
for horizon in [36, 72]:
    plt.figure(figsize=(6.8, 4.2))
    for cap in CAP_ORDER:
        sub = agg[(agg["horizon"] == horizon) & (agg["model_label"] == cap)].sort_values("blocked")
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
    plt.title(f"OOD difficulty grid, horizon={horizon}")
    plt.xticks([1, 2, 3, 4])
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"fig_v33_grid_top1_vs_blocked_h{horizon}.png", dpi=240)
    plt.close()

# Figure 2: same-action degradation for H=72.
plt.figure(figsize=(6.8, 4.2))
for cap in CAP_ORDER:
    sub = agg[(agg["horizon"] == 72) & (agg["model_label"] == cap)].sort_values("blocked")
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
plt.title("State discrimination degrades with constrained geometry, H=72")
plt.xticks([1, 2, 3, 4])
plt.grid(True, alpha=0.25)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_v33_h72_same_action_state_discrimination.png", dpi=240)
plt.close()

print(OUT_MD)
print(OUT_CSV)
print(OUT_AGG)
print()
print(OUT_MD.read_text())

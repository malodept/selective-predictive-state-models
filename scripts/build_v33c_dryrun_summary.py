from __future__ import annotations

from pathlib import Path
import re
import pandas as pd

ROOT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v33_ood_grid")
OUT_CSV = ROOT / "v33c_dryrun_capacity_summary.csv"
OUT_MD = ROOT / "v33c_dryrun_capacity_summary.md"

VARIANTS = ["block1_h36_seed20", "block4_h72_seed20"]
MODELS = ["tiny_full_seed0", "small_full_seed0", "medium_full_seed0", "full_seed0"]
LABELS = {
    "tiny_full_seed0": "Tiny",
    "small_full_seed0": "Small",
    "medium_full_seed0": "Medium",
    "full_seed0": "Full",
}

def parse_md(path: Path) -> dict:
    txt = path.read_text(encoding="utf-8", errors="ignore")
    out = {}
    for line in txt.splitlines():
        m = re.match(r"\|\s*([^|]+?)\s*\|\s*([-+0-9.eE]+)\s*\|", line)
        if not m:
            continue
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

        if "same_state_diff_action" in line:
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            if len(parts) >= 7:
                out["same_state_diff_action_win"] = float(parts[1])
                out["same_state_diff_action_margin"] = float(parts[5])

    return out

rows = []
for variant in VARIANTS:
    for model in MODELS:
        path = ROOT / variant / f"moving_only_{model}.md"
        vals = parse_md(path)
        rows.append({
            "variant": variant,
            "model": model,
            "model_label": LABELS[model],
            **vals,
        })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

lines = []
lines.append("# SPSM v33C dry-run OOD grid summary\n")
lines.append("This summarizes the two dry-run OOD-grid variants before launching the full mini-grid.\n")

lines.append("## Capacity results\n")
lines.append("| variant | Tiny | Small | Medium | Full | best | range |")
lines.append("|---|---:|---:|---:|---:|---|---:|")
for variant in VARIANTS:
    sub = df[df["variant"] == variant].set_index("model_label")
    vals = {m: float(sub.loc[m, "tie_aware_top1"]) for m in ["Tiny", "Small", "Medium", "Full"]}
    best = max(vals, key=vals.get)
    spread = max(vals.values()) - min(vals.values())
    lines.append(
        f"| {variant} | {vals['Tiny']:.6f} | {vals['Small']:.6f} | "
        f"{vals['Medium']:.6f} | {vals['Full']:.6f} | `{best}` | {spread:.6f} |"
    )

lines.append("\n## Decomposition\n")
lines.append("| variant | model | same-action/diff-state | same-state/diff-action |")
lines.append("|---|---|---:|---:|")
for _, r in df.iterrows():
    lines.append(
        f"| {r['variant']} | {r['model_label']} | "
        f"{r.get('same_action_diff_state_win', float('nan')):.6f} | "
        f"{r.get('same_state_diff_action_win', float('nan')):.6f} |"
    )

lines.append("\n## Interpretation\n")
lines.append("- `block1_h36_seed20` is easy and near-saturated.")
lines.append("- `block4_h72_seed20` is valid but hard, with no evidence of massive tie degeneracy.")
lines.append("- This supports launching a systematic blocked/horizon/layout grid.")

OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(OUT_MD)
print(OUT_CSV)
print(OUT_MD.read_text())

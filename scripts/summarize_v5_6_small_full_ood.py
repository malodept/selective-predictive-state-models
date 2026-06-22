from pathlib import Path
import re
import statistics as stats

ROOT_OOD = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_3_ood")
ROOT_SMALL = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_6_small_full")
OUT = ROOT_SMALL / "small_full_ood_summary.md"

VARIANTS = [
    "block2_h72_seed11",
    "block2_v18_seed12",
    "block3_h36_seed13",
    "block3_h48_seed10",
    "block3_h72_seed14",
]

def read(p):
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""

def metric(p, name):
    m = re.search(r"\|\s*" + re.escape(name) + r"\s*\|\s*([0-9.]+)\s*\|", read(p))
    return float(m.group(1)) if m else None

def row_value(p, row_name, col_idx):
    for line in read(p).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == row_name:
            try:
                return float(cells[col_idx])
            except Exception:
                return None
    return None

def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "NA"
    if len(xs) == 1:
        return f"{xs[0]:.6f}"
    return f"{sum(xs)/len(xs):.6f} ± {stats.stdev(xs):.6f}"

lines = []
lines.append("# SPSM v5.6 small-full OOD evaluation\n")
lines.append("The small-full model is a reduced full state-action Transformer: model_dim=128, layers=1, heads=4.")
lines.append("It is compared against the v5.2 full Transformer across controlled OOD variants.\n")

lines.append("| variant | full top-1 mean seeds 0-2 | full seed0 top-1 | small-full seed0 top-1 | gap full seed0 - small | small same-action/diff-state | small same-state/diff-action |")
lines.append("|---|---:|---:|---:|---:|---:|---:|")

for variant in VARIANTS:
    full_vals = []
    for seed in [0, 1, 2]:
        full_vals.append(metric(ROOT_OOD / variant / f"moving_only_full_seed{seed}.md", "strict top-1"))

    full_seed0 = metric(ROOT_OOD / variant / "moving_only_full_seed0.md", "strict top-1")
    small_path = ROOT_SMALL / variant / "moving_only_small_full_seed0.md"
    small = metric(small_path, "strict top-1")

    small_sa = row_value(small_path, "same_action_diff_state", 1)
    small_ss = row_value(small_path, "same_state_diff_action", 1)

    gap = full_seed0 - small if full_seed0 is not None and small is not None else None

    lines.append(
        f"| {variant} | {mean_sd(full_vals)} | {full_seed0:.6f} | {small:.6f} | "
        f"{gap:.6f} | {small_sa:.6f} | {small_ss:.6f} |"
    )

lines.append("\n## Interpretation\n")
lines.append(
    "If small-full remains close to full on easy OOD but drops more on block3 variants, it is a strong cheap/expensive setup for value-of-computation. "
    "If small-full is indistinguishable from full everywhere, then it is already sufficient and the expensive model is not justified in this benchmark."
)

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT)
print(OUT.read_text())

from pathlib import Path
import re
import statistics as stats

ROOT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_3_ood")
OUT = ROOT / "v5_3_ood_variants_summary.md"

def read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def metric(path: Path, name: str):
    if not path.exists():
        return None
    m = re.search(r"\|\s*" + re.escape(name) + r"\s*\|\s*([0-9.]+)\s*\|", read(path))
    return float(m.group(1)) if m else None

def row_metric(path: Path, row_name: str, col_idx: int):
    if not path.exists():
        return None
    for line in read(path).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == row_name:
            try:
                return float(cells[col_idx])
            except Exception:
                return None
    return None

def selective_metric(path: Path, target_coverage: float):
    if not path.exists():
        return None
    for line in read(path).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6:
            try:
                coverage = float(cells[0])
                strict = float(cells[4])
            except ValueError:
                continue
            if abs(coverage - target_coverage) < 1e-9:
                return strict
    return None

def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "NA"
    if len(xs) == 1:
        return f"{xs[0]:.6f}"
    return f"{sum(xs)/len(xs):.6f} ± {stats.stdev(xs):.6f}"

rows = []
for variant in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    moving = []
    same_action = []
    same_state = []
    full_strict = []
    full_tie = []
    ece_tie = []
    cov80 = []
    cov60 = []
    cov50 = []

    for seed in [0, 1, 2]:
        mp = variant / f"moving_only_full_seed{seed}.md"
        cp = variant / f"confidence_full_seed{seed}.md"

        moving.append(metric(mp, "strict top-1"))
        same_action.append(row_metric(mp, "same_action_diff_state", 1))
        same_state.append(row_metric(mp, "same_state_diff_action", 1))

        full_strict.append(metric(cp, "strict top-1"))
        full_tie.append(metric(cp, "tie-aware top-1"))
        ece_tie.append(metric(cp, "ECE vs tie-aware target"))

        cov80.append(selective_metric(cp, 0.80))
        cov60.append(selective_metric(cp, 0.60))
        cov50.append(selective_metric(cp, 0.50))

    rows.append({
        "variant": variant.name,
        "moving": mean_sd(moving),
        "same_action": mean_sd(same_action),
        "same_state": mean_sd(same_state),
        "full_strict": mean_sd(full_strict),
        "full_tie": mean_sd(full_tie),
        "ece_tie": mean_sd(ece_tie),
        "cov80": mean_sd(cov80),
        "cov60": mean_sd(cov60),
        "cov50": mean_sd(cov50),
    })

lines = []
lines.append("# SPSM v5.3 OOD variants summary\n")
lines.append("| variant | moving-only strict top-1 | same-action/diff-state | same-state/diff-action | full strict top-1 | full tie-aware top-1 | ECE tie-aware | strict @80% cov | strict @60% cov | strict @50% cov |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for r in rows:
    lines.append(
        f"| {r['variant']} | {r['moving']} | {r['same_action']} | {r['same_state']} | "
        f"{r['full_strict']} | {r['full_tie']} | {r['ece_tie']} | "
        f"{r['cov80']} | {r['cov60']} | {r['cov50']} |"
    )

lines.append("\n## Main interpretation\n")
lines.append(
    "The frozen v5.2 state-action Transformer remains highly robust to horizon-only and velocity-only shifts, "
    "but degrades substantially when the number of blocked directions increases from 2 to 3. "
    "The dominant failure mode is same-action/different-state discrimination, while same-state/different-action discrimination remains much stronger. "
    "This suggests that the model keeps action grounding but struggles with more constrained environment geometry. "
    "Confidence remains useful for selective prediction: high-confidence subsets recover much stronger strict top-1 under OOD shift."
)

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT)
print(OUT.read_text())

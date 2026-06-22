from pathlib import Path
import re

ROOT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v0_paper_id_ablation")
OUT = Path("paper/spsm_v0/table_id_ablation.tex")

MODES = [
    ("full", "Full"),
    ("action_only", "Action-only"),
    ("state_only", "State-only"),
    ("no_context", "No-context"),
]

def read(path):
    return path.read_text(encoding="utf-8", errors="replace")

def metric(path, name):
    txt = read(path)
    m = re.search(r"\|\s*" + re.escape(name) + r"\s*\|\s*([0-9.]+)\s*\|", txt)
    if not m:
        return None
    return float(m.group(1))

def row_value(path, row_name, col_idx):
    for line in read(path).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == row_name:
            return float(cells[col_idx])
    return None

def fmt(x):
    if x is None:
        return "--"
    return f"{x:.3f}"

rows = []
for mode, label in MODES:
    path = ROOT / f"moving_only_{mode}_seed0.md"
    rows.append([
        label,
        fmt(metric(path, "strict top-1")),
        fmt(row_value(path, "same_action_diff_state", 1)),
        fmt(row_value(path, "same_state_diff_action", 1)),
    ])

lines = []
lines.append(r"\begin{table}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(r"\begin{tabular}{lrrr}")
lines.append(r"\toprule")
lines.append(r"Model & Top-1 & Same-action & Same-state \\")
lines.append(r"\midrule")
for r in rows:
    lines.append(" & ".join(r) + r" \\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\caption{ID moving-only ablation study under the state-disjoint exact-intervention protocol. The full state-action model combines both discrimination axes, while incomplete-context ablations capture only partial structure. Same-action denotes same-action/different-state negatives; same-state denotes same-state/different-action negatives.}")
lines.append(r"\label{tab:id_ablation}")
lines.append(r"\end{table}")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT)
print(OUT.read_text())

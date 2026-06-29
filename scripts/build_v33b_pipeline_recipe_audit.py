from __future__ import annotations

from pathlib import Path
import re

OUT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v33_ood_grid/v33b_pipeline_recipe_audit.md")
OUT.parent.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    "scripts/create_pybullet_obstacle_intervention_rgb_dataset.py",
    "scripts/extract_pybullet_rgb_dinov2_features.py",
    "scripts/pool_dinov2_patchtokens.py",
    "scripts/mine_state_disjoint_mixed_hard_groups.py",
    "scripts/eval_mixed_hard_tieaware_moving_only.py",
    "scripts/eval_state_disjoint_confidence_diagnostics.py",
]

LOG_PATTERNS = [
    "logs/v5_3_ood/create_*.log",
    "logs/v5_3_ood/extract_dinov2_*.log",
    "logs/v5_3_ood/pool_*.log",
    "logs/v5_3_ood/mine_*.log",
]

def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

lines = []
lines.append("# SPSM v33B pipeline recipe audit\n")
lines.append("This audit extracts the existing v5_3_ood generation recipe before expanding to a systematic OOD grid.\n")

lines.append("## Script argument definitions\n")
for sp in SCRIPTS:
    p = Path(sp)
    lines.append(f"\n### `{sp}`\n")
    if not p.exists():
        lines.append("MISSING")
        continue

    txt = read(p)
    keep = []
    for i, line in enumerate(txt.splitlines(), start=1):
        if "ArgumentParser" in line or "add_argument" in line:
            keep.append(f"{i}: {line.rstrip()}")

    lines.append("```")
    lines.extend(keep[:220])
    lines.append("```")

lines.append("\n## Existing v5_3_ood logs: first/last useful lines\n")
for pat in LOG_PATTERNS:
    for lp in sorted(Path(".").glob(pat)):
        txt = read(lp)
        raw = txt.splitlines()
        useful = [
            x for x in raw
            if (
                "python" in x
                or "phipy" in x
                or "--" in x
                or "Saved" in x
                or "Wrote" in x
                or "OUT" in x
                or "shape" in x
                or "horizon" in x
                or "velocity" in x
                or "blocked" in x
                or "DONE" in x
            )
        ]

        lines.append(f"\n### `{lp}`\n")
        lines.append("```")
        if useful:
            lines.extend(useful[:80])
            if len(useful) > 120:
                lines.append("...")
            lines.extend(useful[-40:])
        else:
            lines.extend(raw[:40])
            if len(raw) > 80:
                lines.append("...")
            lines.extend(raw[-40:])
        lines.append("```")

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(OUT)
print(OUT.read_text())

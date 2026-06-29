from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("outputs/counterfactual/pybullet_obstacles_rgb_v33_ood_grid")
OUT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v33_ood_grid")
OUT_MD = OUT / "v33g_block4_sanity_check.md"
OUT_CSV = OUT / "v33g_block4_sanity_check.csv"

rows = []

for seed in [20, 21, 22]:
    p36 = BASE / f"block4_h36_seed{seed}" / f"pybullet_obstacle_rgb_v1_ood_block4_h36_seed{seed}_dinov2_vits14_pool4.npz"
    p72 = BASE / f"block4_h72_seed{seed}" / f"pybullet_obstacle_rgb_v1_ood_block4_h72_seed{seed}_dinov2_vits14_pool4.npz"

    a = np.load(p36, allow_pickle=True)
    b = np.load(p72, allow_pickle=True)

    row = {"seed": seed}

    for key in ["z_current", "z_future", "action", "action_id", "group_id", "state_xy", "future_xy", "blocked_mask"]:
        if key in a and key in b:
            aa = a[key]
            bb = b[key]
            row[f"{key}_same_shape"] = aa.shape == bb.shape
            if np.issubdtype(aa.dtype, np.number):
                row[f"{key}_max_abs_diff"] = float(np.max(np.abs(aa.astype(np.float64) - bb.astype(np.float64))))
                row[f"{key}_allclose"] = bool(np.allclose(aa, bb))
            else:
                row[f"{key}_equal"] = bool(np.array_equal(aa, bb))

    for key in ["horizon", "velocity", "num_blocked_directions", "dataset_version"]:
        if key in a and key in b:
            row[f"{key}_h36"] = str(a[key].item() if a[key].shape == () else a[key])
            row[f"{key}_h72"] = str(b[key].item() if b[key].shape == () else b[key])

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

lines = []
lines.append("# v33G block4 horizon sanity check\n")
lines.append("This checks whether block4 H=36 and H=72 are genuinely identical or only identical at the evaluation level.\n")
lines.append("| seed | z_current diff | z_future diff | future_xy diff | action diff | horizon values |")
lines.append("|---:|---:|---:|---:|---:|---|")

for _, r in df.iterrows():
    lines.append(
        f"| {int(r['seed'])} | "
        f"{r.get('z_current_max_abs_diff', float('nan')):.6g} | "
        f"{r.get('z_future_max_abs_diff', float('nan')):.6g} | "
        f"{r.get('future_xy_max_abs_diff', float('nan')):.6g} | "
        f"{r.get('action_max_abs_diff', float('nan')):.6g} | "
        f"{r.get('horizon_h36', '')} / {r.get('horizon_h72', '')} |"
    )

lines.append("\n## Interpretation")
lines.append("- If future states/features are identical while horizon metadata differs, blocked=4 makes movement fully constrained, so horizon becomes physically irrelevant.")
lines.append("- If files differ but metrics match, the equality is an evaluation-level coincidence.")
lines.append("- If paths or metadata are wrong, fix v33 before using it in the paper.")

OUT_MD.write_text('\\n'.join(lines) + '\\n', encoding='utf-8')

print(OUT_MD)
print(OUT_CSV)
print(OUT_MD.read_text())

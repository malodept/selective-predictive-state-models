from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("outputs/counterfactual/pybullet_obstacles_rgb_v33_ood_grid")
OUT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v33_ood_grid")
rows = []

for seed in [20, 21, 22]:
    p36 = BASE / f"block4_h36_seed{seed}" / f"pybullet_obstacle_rgb_v1_ood_block4_h36_seed{seed}.npz"
    p72 = BASE / f"block4_h72_seed{seed}" / f"pybullet_obstacle_rgb_v1_ood_block4_h72_seed{seed}.npz"

    a = np.load(p36, allow_pickle=True)
    b = np.load(p72, allow_pickle=True)

    row = {"seed": seed}
    for key in ["current_rgb", "future_rgb", "state_xy", "future_xy", "action", "action_id", "blocked_mask"]:
        if key in a and key in b:
            aa, bb = a[key], b[key]
            row[f"{key}_same_shape"] = aa.shape == bb.shape
            if np.issubdtype(aa.dtype, np.number):
                diff = np.abs(aa.astype(np.float64) - bb.astype(np.float64))
                row[f"{key}_max_abs_diff"] = float(diff.max())
                row[f"{key}_mean_abs_diff"] = float(diff.mean())
                row[f"{key}_allclose"] = bool(np.allclose(aa, bb))
            else:
                row[f"{key}_equal"] = bool(np.array_equal(aa, bb))

    rows.append(row)

df = pd.DataFrame(rows)
csv = OUT / "v33h_block4_raw_rgb_check.csv"
md = OUT / "v33h_block4_raw_rgb_check.md"
df.to_csv(csv, index=False)

lines = ["# v33H block4 raw RGB sanity check\n"]
lines.append("| seed | current RGB max diff | future RGB max diff | future XY max diff | action diff |")
lines.append("|---:|---:|---:|---:|---:|")
for _, r in df.iterrows():
    lines.append(
        f"| {int(r['seed'])} | "
        f"{r.get('current_rgb_max_abs_diff', float('nan')):.6g} | "
        f"{r.get('future_rgb_max_abs_diff', float('nan')):.6g} | "
        f"{r.get('future_xy_max_abs_diff', float('nan')):.6g} | "
        f"{r.get('action_max_abs_diff', float('nan')):.6g} |"
    )

lines.append("\n## Interpretation")
lines.append("- If future RGB differs while future_xy is nearly identical, the equality of metrics is not file reuse; it is a physically similar but visually non-bitwise-identical endpoint.")
lines.append("- If future RGB is identical too, then H=36/H=72 are effectively identical in blocked=4.")
md.write_text("\n".join(lines) + "\n")

print(md)
print(csv)
print(md.read_text())

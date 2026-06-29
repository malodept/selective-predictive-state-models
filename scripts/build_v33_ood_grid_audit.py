from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v33_ood_grid")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MD = OUT_DIR / "v33_ood_grid_audit.md"
OUT_CSV = OUT_DIR / "v33_existing_ood_npz_inventory.csv"

ROOTS = [
    Path("outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood"),
    Path("outputs/counterfactual/pybullet_obstacles_rgb_scale"),
]

rows = []

for root in ROOTS:
    if not root.exists():
        continue

    for npz_path in sorted(root.rglob("*.npz")):
        name = str(npz_path)
        if "dinov2" not in name and "groups" not in name:
            continue

        row = {
            "path": name,
            "size_mb": npz_path.stat().st_size / 1024**2,
            "parent": npz_path.parent.name,
            "is_groups": "groups" in npz_path.name,
            "is_dinov2": "dinov2" in npz_path.name,
        }

        try:
            z = np.load(npz_path, allow_pickle=True)
            row["keys"] = ",".join(list(z.keys())[:30])

            for k in [
                "z_current",
                "z_future",
                "candidate_indices",
                "correct_candidate",
                "action",
                "action_id",
                "group_id",
                "anchor_indices",
                "negative_type",
                "split",
                "horizon",
                "velocity",
                "num_blocked_directions",
                "dataset_version",
                "image_size",
            ]:
                if k in z:
                    arr = z[k]
                    row[f"{k}_shape"] = str(getattr(arr, "shape", "scalar"))
                    if getattr(arr, "shape", ()) == ():
                        try:
                            row[f"{k}_value"] = str(arr.item())
                        except Exception:
                            row[f"{k}_value"] = str(arr)
        except Exception as e:
            row["error"] = repr(e)

        rows.append(row)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)

lines = []
lines.append("# SPSM v33 OOD grid audit\n")
lines.append("This audit inventories existing OOD datasets and group files before expanding the controlled OOD grid.\n")

lines.append("## Existing NPZ inventory\n")
lines.append(f"- files found: `{len(df)}`")
if len(df):
    lines.append(f"- total size GB: `{df['size_mb'].sum()/1024:.3f}`")
    lines.append("")

    show = df[["parent", "path", "size_mb", "is_groups", "is_dinov2"]].copy()
    lines.append("| parent | file | MB | groups? | dinov2? |")
    lines.append("|---|---|---:|---|---|")
    for _, r in show.iterrows():
        lines.append(
            f"| {r['parent']} | `{Path(r['path']).name}` | {r['size_mb']:.2f} | "
            f"{bool(r['is_groups'])} | {bool(r['is_dinov2'])} |"
        )

lines.append("\n## Existing scalar metadata\n")
meta_cols = [c for c in df.columns if c.endswith("_value")]
if meta_cols:
    cols = ["parent", "path"] + meta_cols
    lines.append("| parent | file | " + " | ".join(meta_cols) + " |")
    lines.append("|---|---|" + "|".join(["---"] * len(meta_cols)) + "|")
    for _, r in df[cols].iterrows():
        vals = []
        for c in meta_cols:
            vals.append(str(r.get(c, "")) if pd.notna(r.get(c, "")) else "")
        lines.append(f"| {r['parent']} | `{Path(r['path']).name}` | " + " | ".join(vals) + " |")

lines.append("\n## Proposed v33 goal\n")
lines.append("The next experiment should expand from a few named OOD variants to a systematic grid over geometry, horizon, velocity, and layout seed.")
lines.append("The first priority is to identify the correct dataset-generation script and its CLI flags from the code-context audit.")

OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(OUT_MD)
print(OUT_CSV)
print(OUT_MD.read_text())

#!/usr/bin/env bash
set -euo pipefail

cd /shared/home/mdepastor/projects/spsm

mkdir -p logs/v18_multiseed_small_full_ood
mkdir -p reports/tables/protocol/pybullet_obstacle_rgb_encoder/v18_multiseed_small_full_ood

for SEED in 1 2; do
  SRC="scripts/export_v5_6_small_full_voc_examples.py"
  TMP="scripts/export_v18_small_full_voc_examples_seed${SEED}.py"
  OUT="reports/tables/protocol/pybullet_obstacle_rgb_encoder/v18_multiseed_small_full_ood/small_full_voc_examples_seed${SEED}.csv"

  cp "$SRC" "$TMP"

  phipy - <<PY
from pathlib import Path

seed = ${SEED}
p = Path("${TMP}")
s = p.read_text()

old_ckpt = "outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed0/checkpoint.pt"
new_ckpt = f"outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed{seed}/checkpoint.pt"

old_out = "reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_6_small_full/small_full_voc_examples.csv"
new_out = f"reports/tables/protocol/pybullet_obstacle_rgb_encoder/v18_multiseed_small_full_ood/small_full_voc_examples_seed{seed}.csv"

if old_ckpt not in s:
    raise SystemExit("small_full seed0 checkpoint string not found")
if old_out not in s:
    raise SystemExit("output CSV string not found")

s = s.replace(old_ckpt, new_ckpt)
s = s.replace(old_out, new_out)

p.write_text(s)
print("patched", p)
PY

  echo "======================================================================"
  echo "Exporting small_full VoC examples seed=${SEED}"
  echo "======================================================================"

  phipy -u "$TMP" \
    > "logs/v18_multiseed_small_full_ood/export_small_full_voc_seed${SEED}.log" 2>&1

  phipy - <<PY
from pathlib import Path
import pandas as pd

seed = ${SEED}
p = Path("${OUT}")

df = pd.read_csv(p)

if "cheap_seed" not in df.columns:
    df.insert(0, "cheap_seed", seed)

df.to_csv(p, index=False)

print(p)
print("shape", df.shape)
print(df["mode"].value_counts().to_string())
print(df["variant"].value_counts().to_string())
PY

done

echo "DONE v18 VoC export"

#!/usr/bin/env bash
set -euo pipefail

CONTAINER="/shared/projects/phisat2/containers/phisat2.sif"

NAME="block3_h48_seed10"
ROOT_OUT="outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/${NAME}"
ROOT_REP="reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_3_ood/${NAME}"
LOG_DIR="logs/v5_3_ood"
mkdir -p "${ROOT_OUT}" "${ROOT_REP}" "${LOG_DIR}"

DATA_POOL="${ROOT_OUT}/pybullet_obstacle_rgb_v1_ood_${NAME}_dinov2_vits14_pool4.npz"
GROUP_FILE="${ROOT_OUT}/mixed_hard_state_disjoint_v0_groups.npz"

echo "===== RESUME V5.3 OOD PILOT EVAL: ${NAME} ====="

echo
echo "===== 0. Check pooled feature file ====="
if [ ! -f "${DATA_POOL}" ]; then
  echo "MISSING pooled feature file: ${DATA_POOL}"
  exit 1
fi
ls -lh "${DATA_POOL}"

echo
echo "===== 1. Recover or regenerate group file ====="
if [ -f "${GROUP_FILE}" ]; then
  echo "Group file already exists:"
  ls -lh "${GROUP_FILE}"
elif [ -f "10000.npz" ]; then
  echo "Recovering accidental 10000.npz group file"
  mv -v "10000.npz" "${GROUP_FILE}"
elif [ -f "10000" ]; then
  echo "Recovering accidental 10000 group file"
  mv -v "10000" "${GROUP_FILE}"
else
  echo "Group file not found; regenerating mining only"
  apptainer exec --nv "${CONTAINER}" \
    python -u scripts/mine_state_disjoint_mixed_hard_groups.py \
      --data "${DATA_POOL}" \
      --out "${GROUP_FILE}" \
      --report "${ROOT_REP}/mixed_hard_state_disjoint_groups.json" \
      --train-groups 3000 \
      --val-groups 500 \
      --test-groups 500 \
      --same-state-negatives 2 \
      --same-action-negatives 2 \
      --max-pos-dist 0.15 \
      --prefer-block-mismatch \
      --seed 10 \
    2>&1 | tee "${LOG_DIR}/mine_${NAME}_resume.log"
fi

echo
echo "===== 2. Audit group file ====="
apptainer exec --nv "${CONTAINER}" python - <<PY
import numpy as np
p = "${GROUP_FILE}"
g = np.load(p, allow_pickle=True)
print("group_file:", p)
print("keys:", list(g.files))
for k in g.files:
    arr = g[k]
    if hasattr(arr, "shape"):
        print(k, arr.shape, arr.dtype)
PY

echo
echo "===== 3. Evaluate v5.2 full Transformer checkpoints on OOD moving-only ====="
for SEED in 0 1 2; do
  CKPT="outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed${SEED}/checkpoint.pt"
  OUT_JSON="${ROOT_REP}/moving_only_full_seed${SEED}.json"

  if [ ! -f "${CKPT}" ]; then
    echo "MISSING checkpoint: ${CKPT}"
    exit 1
  fi

  apptainer exec --nv "${CONTAINER}" \
    python -u scripts/eval_mixed_hard_tieaware_moving_only.py \
      --data "${DATA_POOL}" \
      --groups "${GROUP_FILE}" \
      --checkpoint "${CKPT}" \
      --out "${OUT_JSON}" \
      --batch-groups 128 \
      --device cuda \
    2>&1 | tee "${LOG_DIR}/moving_only_full_seed${SEED}_${NAME}_resume.log"
done

echo
echo "===== 4. Evaluate v5.2 confidence diagnostics on OOD ====="
for SEED in 0 1 2; do
  CKPT="outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed${SEED}/checkpoint.pt"
  OUT_JSON="${ROOT_REP}/confidence_full_seed${SEED}.json"

  apptainer exec --nv "${CONTAINER}" \
    python -u scripts/eval_state_disjoint_confidence_diagnostics.py \
      --data "${DATA_POOL}" \
      --groups "${GROUP_FILE}" \
      --checkpoint "${CKPT}" \
      --out "${OUT_JSON}" \
      --temperature 0.02 \
      --batch-groups 128 \
      --device cuda \
    2>&1 | tee "${LOG_DIR}/confidence_full_seed${SEED}_${NAME}_resume.log"
done

echo
echo "===== 5. Robust JSON summary ====="
python - <<'PY'
from pathlib import Path
import json
import math

root = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_3_ood/block3_h48_seed10")

def flatten(x, prefix=""):
    rows = []
    if isinstance(x, dict):
        for k, v in x.items():
            rows += flatten(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(x, list):
        if len(x) <= 10 and all(isinstance(v, (int, float, str, bool, type(None))) for v in x):
            rows.append((prefix, x))
    elif isinstance(x, (int, float, str, bool)) or x is None:
        rows.append((prefix, x))
    return rows

for kind in ["moving_only", "confidence"]:
    print(f"\n## {kind}")
    for seed in [0, 1, 2]:
        path = root / f"{kind}_full_seed{seed}.json"
        print(f"\n### seed={seed}")
        if not path.exists():
            print("MISSING", path)
            continue
        data = json.loads(path.read_text())
        for k, v in flatten(data):
            print(f"{k}: {v}")
PY

echo
echo "DONE: resumed ${NAME}"

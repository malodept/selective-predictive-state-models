#!/usr/bin/env bash
set -euo pipefail

VARIANTS=(
  block2_h72_seed11
  block2_v18_seed12
  block3_h36_seed13
  block3_h48_seed10
  block3_h72_seed14
)

CKPT="outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed0/checkpoint.pt"

if [ ! -f "${CKPT}" ]; then
  echo "MISSING CHECKPOINT: ${CKPT}"
  exit 1
fi

for VARIANT in "${VARIANTS[@]}"; do
  echo "===== SMALL FULL OOD EVAL: ${VARIANT} ====="

  DATA_FILE="outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/${VARIANT}/pybullet_obstacle_rgb_v1_ood_${VARIANT}_dinov2_vits14_pool4.npz"
  GROUP_FILE="outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/${VARIANT}/mixed_hard_state_disjoint_v0_groups.npz"
  OUT_DIR="reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_6_small_full/${VARIANT}"
  LOG_DIR="logs/v5_6_small_full"
  mkdir -p "${OUT_DIR}" "${LOG_DIR}"

  phipy scripts/eval_mixed_hard_tieaware_moving_only.py \
    --data "${DATA_FILE}" \
    --groups "${GROUP_FILE}" \
    --checkpoint "${CKPT}" \
    --out "${OUT_DIR}/moving_only_small_full_seed0.json" \
    --batch-groups 128 \
    --device cuda \
    2>&1 | tee "${LOG_DIR}/moving_only_small_full_${VARIANT}.log"

  phipy scripts/eval_state_disjoint_confidence_diagnostics.py \
    --data "${DATA_FILE}" \
    --groups "${GROUP_FILE}" \
    --checkpoint "${CKPT}" \
    --out "${OUT_DIR}/confidence_small_full_seed0.json" \
    --temperature 0.02 \
    --batch-groups 128 \
    --device cuda \
    2>&1 | tee "${LOG_DIR}/confidence_small_full_${VARIANT}.log"
done

echo "DONE small_full OOD eval"

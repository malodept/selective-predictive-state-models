#!/usr/bin/env bash
set -euo pipefail

VARIANTS=(
  block2_h72_seed11
  block2_v18_seed12
  block3_h36_seed13
  block3_h48_seed10
  block3_h72_seed14
)

MODES=(
  action_only
  state_only
  no_context
)

for VARIANT in "${VARIANTS[@]}"; do
  echo "===== OOD ABLATION EVAL: ${VARIANT} ====="

  DATA_FILE="outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/${VARIANT}/pybullet_obstacle_rgb_v1_ood_${VARIANT}_dinov2_vits14_pool4.npz"
  GROUP_FILE="outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/${VARIANT}/mixed_hard_state_disjoint_v0_groups.npz"
  OUT_DIR="reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_3_ood/${VARIANT}/ablations"
  LOG_DIR="logs/v5_3_ood"
  mkdir -p "${OUT_DIR}" "${LOG_DIR}"

  if [ ! -f "${DATA_FILE}" ]; then
    echo "MISSING DATA: ${DATA_FILE}"
    exit 1
  fi

  if [ ! -f "${GROUP_FILE}" ]; then
    echo "MISSING GROUP FILE: ${GROUP_FILE}"
    exit 1
  fi

  for MODE in "${MODES[@]}"; do
    CKPT="outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/${MODE}_seed0/checkpoint.pt"

    if [ ! -f "${CKPT}" ]; then
      echo "MISSING CHECKPOINT: ${CKPT}"
      exit 1
    fi

    echo "----- ${VARIANT} / ${MODE} -----"

    phipy scripts/eval_mixed_hard_tieaware_moving_only.py \
      --data "${DATA_FILE}" \
      --groups "${GROUP_FILE}" \
      --checkpoint "${CKPT}" \
      --out "${OUT_DIR}/moving_only_${MODE}_seed0.json" \
      --batch-groups 128 \
      --device cuda \
      2>&1 | tee "${LOG_DIR}/moving_only_${MODE}_${VARIANT}.log"
  done
done

echo "DONE OOD ablation eval"

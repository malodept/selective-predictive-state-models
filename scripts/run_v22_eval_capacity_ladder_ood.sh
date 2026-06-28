#!/usr/bin/env bash
set -euo pipefail

cd /shared/home/mdepastor/projects/spsm

mkdir -p logs/v22_capacity_ladder_ood
mkdir -p reports/tables/protocol/pybullet_obstacle_rgb_encoder/v22_capacity_ladder_ood

VARIANTS=(
  block2_h72_seed11
  block2_v18_seed12
  block3_h36_seed13
  block3_h48_seed10
  block3_h72_seed14
)

MODELS=(
  tiny_full_seed0
  small_full_seed0
  medium_full_seed0
  full_seed0
)

for MODEL in "${MODELS[@]}"; do
  CKPT="outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/${MODEL}/checkpoint.pt"

  if [ ! -f "$CKPT" ]; then
    echo "[MISSING CKPT] $CKPT"
    exit 1
  fi

  for VARIANT in "${VARIANTS[@]}"; do
    DATA="outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/${VARIANT}/pybullet_obstacle_rgb_v1_ood_${VARIANT}_dinov2_vits14_pool4.npz"
    GROUPS_FILE="outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/${VARIANT}/mixed_hard_state_disjoint_v0_groups.npz"

    OUT_DIR="reports/tables/protocol/pybullet_obstacle_rgb_encoder/v22_capacity_ladder_ood/${VARIANT}"
    mkdir -p "$OUT_DIR"

    echo "======================================================================"
    echo "model=${MODEL} variant=${VARIANT}"
    echo "======================================================================"

    if [ ! -f "${OUT_DIR}/moving_only_${MODEL}.json" ]; then
      phipy -u scripts/eval_mixed_hard_tieaware_moving_only.py \
        --data "$DATA" \
        --groups "$GROUPS_FILE" \
        --checkpoint "$CKPT" \
        --out "${OUT_DIR}/moving_only_${MODEL}.json" \
        --batch-groups 128 \
        --device cuda \
        2>&1 | tee "logs/v22_capacity_ladder_ood/moving_only_${MODEL}_${VARIANT}.log"
    else
      echo "[SKIP] moving-only exists"
    fi

    if [ ! -f "${OUT_DIR}/confidence_${MODEL}.json" ]; then
      phipy -u scripts/eval_state_disjoint_confidence_diagnostics.py \
        --data "$DATA" \
        --groups "$GROUPS_FILE" \
        --checkpoint "$CKPT" \
        --out "${OUT_DIR}/confidence_${MODEL}.json" \
        --temperature 0.02 \
        --batch-groups 128 \
        --device cuda \
        2>&1 | tee "logs/v22_capacity_ladder_ood/confidence_${MODEL}_${VARIANT}.log"
    else
      echo "[SKIP] confidence exists"
    fi
  done
done

echo "DONE v22 capacity ladder OOD eval"

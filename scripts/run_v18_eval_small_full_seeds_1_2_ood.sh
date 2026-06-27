#!/usr/bin/env bash
set -euo pipefail

cd /shared/home/mdepastor/projects/spsm

mkdir -p logs/v18_multiseed_small_full_ood
mkdir -p reports/tables/protocol/pybullet_obstacle_rgb_encoder/v18_multiseed_small_full_ood

VARIANTS=(
  block2_h72_seed11
  block2_v18_seed12
  block3_h36_seed13
  block3_h48_seed10
  block3_h72_seed14
)

for SEED in 1 2; do
  CKPT="outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed${SEED}/checkpoint.pt"

  if [ ! -f "$CKPT" ]; then
    echo "[MISSING CKPT] $CKPT"
    exit 1
  fi

  for VARIANT in "${VARIANTS[@]}"; do
    DATA="outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/${VARIANT}/pybullet_obstacle_rgb_v1_ood_${VARIANT}_dinov2_vits14_pool4.npz"
    GROUPS_FILE="outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/${VARIANT}/mixed_hard_state_disjoint_v0_groups.npz"

    OUT_DIR="reports/tables/protocol/pybullet_obstacle_rgb_encoder/v18_multiseed_small_full_ood/${VARIANT}"
    mkdir -p "$OUT_DIR"

    echo "======================================================================"
    echo "seed=${SEED} variant=${VARIANT}"
    echo "======================================================================"

    if [ ! -f "${OUT_DIR}/moving_only_small_full_seed${SEED}.json" ]; then
      phipy -u scripts/eval_mixed_hard_tieaware_moving_only.py \
        --data "$DATA" \
        --groups "$GROUPS_FILE" \
        --checkpoint "$CKPT" \
        --out "${OUT_DIR}/moving_only_small_full_seed${SEED}.json" \
        --batch-groups 128 \
        --device cuda \
        2>&1 | tee "logs/v18_multiseed_small_full_ood/moving_only_small_full_seed${SEED}_${VARIANT}.log"
    else
      echo "[SKIP] moving-only exists"
    fi

    if [ ! -f "${OUT_DIR}/confidence_small_full_seed${SEED}.json" ]; then
      phipy -u scripts/eval_state_disjoint_confidence_diagnostics.py \
        --data "$DATA" \
        --groups "$GROUPS_FILE" \
        --checkpoint "$CKPT" \
        --out "${OUT_DIR}/confidence_small_full_seed${SEED}.json" \
        --temperature 0.02 \
        --batch-groups 128 \
        --device cuda \
        2>&1 | tee "logs/v18_multiseed_small_full_ood/confidence_small_full_seed${SEED}_${VARIANT}.log"
    else
      echo "[SKIP] confidence exists"
    fi
  done
done

echo "DONE v18 OOD eval"

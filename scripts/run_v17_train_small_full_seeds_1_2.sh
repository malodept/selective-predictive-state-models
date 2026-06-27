#!/usr/bin/env bash
set -euo pipefail

cd /shared/home/mdepastor/projects/spsm

mkdir -p logs/v17_multiseed_small_full
mkdir -p reports/tables/protocol/pybullet_obstacle_rgb_encoder/v17_multiseed_small_full

DATA="outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz"
GROUPS_FILE="outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz"

for SEED in 1 2; do
  OUT_DIR="outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed${SEED}"
  REPORT="reports/tables/protocol/pybullet_obstacle_rgb_encoder/v17_multiseed_small_full/small_full_seed${SEED}.json"

  if [ -f "${OUT_DIR}/checkpoint.pt" ]; then
    echo "[SKIP] ${OUT_DIR}/checkpoint.pt already exists"
    continue
  fi

  echo "======================================================================"
  echo "Training small_full seed=${SEED}"
  echo "======================================================================"

  phipy -u scripts/train_mixed_hard_delta_transformer.py \
    --data "${DATA}" \
    --groups "${GROUPS_FILE}" \
    --out-dir "${OUT_DIR}" \
    --report "${REPORT}" \
    --mode full \
    --epochs 100 \
    --batch-groups 64 \
    --eval-batch-groups 64 \
    --lr 0.0003 \
    --weight-decay 0.0001 \
    --model-dim 128 \
    --heads 4 \
    --layers 1 \
    --dropout 0.05 \
    --temperature 0.02 \
    --lambda-mse 0.01 \
    --device cuda \
    --seed "${SEED}"
done

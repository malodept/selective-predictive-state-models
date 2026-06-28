#!/usr/bin/env bash
set -euo pipefail

cd /shared/home/mdepastor/projects/spsm

mkdir -p logs/v21_capacity_ladder
mkdir -p reports/tables/protocol/pybullet_obstacle_rgb_encoder/v21_capacity_ladder

DATA="outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz"
GROUPS_FILE="outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz"

run_model () {
  NAME="$1"
  MODEL_DIM="$2"
  LAYERS="$3"
  HEADS="$4"
  SEED="$5"

  OUT_DIR="outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/${NAME}_seed${SEED}"
  REPORT="reports/tables/protocol/pybullet_obstacle_rgb_encoder/v21_capacity_ladder/${NAME}_seed${SEED}.json"

  if [ -f "${OUT_DIR}/checkpoint.pt" ]; then
    echo "[SKIP] ${OUT_DIR}/checkpoint.pt already exists"
    return
  fi

  echo "======================================================================"
  echo "Training ${NAME}_seed${SEED}: dim=${MODEL_DIM}, layers=${LAYERS}, heads=${HEADS}"
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
    --model-dim "${MODEL_DIM}" \
    --heads "${HEADS}" \
    --layers "${LAYERS}" \
    --dropout 0.05 \
    --temperature 0.02 \
    --lambda-mse 0.01 \
    --device cuda \
    --seed "${SEED}"
}

run_model tiny_full 64 1 4 0
run_model medium_full 192 2 4 0

echo "DONE v21 capacity ladder training"

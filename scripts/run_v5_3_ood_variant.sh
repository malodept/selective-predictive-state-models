#!/usr/bin/env bash
set -euo pipefail

NAME="$1"
GROUPS_N="$2"
HORIZON="$3"
VELOCITY="$4"
BLOCKED="$5"
SEED="$6"

CONTAINER="/shared/projects/phisat2/containers/phisat2.sif"

ROOT_OUT="outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/${NAME}"
ROOT_REP="reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_3_ood/${NAME}"
LOG_DIR="logs/v5_3_ood"
mkdir -p "${ROOT_OUT}" "${ROOT_REP}" "${LOG_DIR}"

DATA_RGB="${ROOT_OUT}/pybullet_obstacle_rgb_v1_ood_${NAME}.npz"
DATA_DINO="${ROOT_OUT}/pybullet_obstacle_rgb_v1_ood_${NAME}_dinov2_vits14.npz"
DATA_POOL="${ROOT_OUT}/pybullet_obstacle_rgb_v1_ood_${NAME}_dinov2_vits14_pool4.npz"
GROUP_FILE="${ROOT_OUT}/mixed_hard_state_disjoint_v0_groups.npz"

echo "===== V5.3 OOD VARIANT: ${NAME} ====="
echo "groups=${GROUPS_N} horizon=${HORIZON} velocity=${VELOCITY} blocked=${BLOCKED} seed=${SEED}"

if [ ! -f "${DATA_RGB}" ]; then
  apptainer exec --nv "${CONTAINER}" \
    python -u scripts/create_pybullet_obstacle_intervention_rgb_dataset.py \
      --out "${DATA_RGB}" \
      --report "${ROOT_REP}/dataset.json" \
      --groups "${GROUPS_N}" \
      --image-size 96 \
      --patch-grid 4 \
      --horizon "${HORIZON}" \
      --velocity "${VELOCITY}" \
      --num-blocked-directions "${BLOCKED}" \
      --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/create_${NAME}.log"
fi

if [ ! -f "${DATA_DINO}" ]; then
  apptainer exec --nv "${CONTAINER}" \
    python -u scripts/extract_pybullet_rgb_dinov2_features.py \
      --input "${DATA_RGB}" \
      --output "${DATA_DINO}" \
      --report "${ROOT_REP}/dinov2_vits14.json" \
      --model dinov2_vits14 \
      --image-size 224 \
      --batch-size 64 \
      --device cuda \
    2>&1 | tee "${LOG_DIR}/extract_dinov2_${NAME}.log"
fi

if [ ! -f "${DATA_POOL}" ]; then
  apptainer exec --nv "${CONTAINER}" \
    python -u scripts/pool_dinov2_patchtokens.py \
      --input "${DATA_DINO}" \
      --output "${DATA_POOL}" \
      --report "${ROOT_REP}/dinov2_vits14_pool4.json" \
      --source-grid 16 \
      --target-grid 4 \
    2>&1 | tee "${LOG_DIR}/pool_${NAME}.log"
fi

if [ ! -f "${GROUP_FILE}" ]; then
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
      --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/mine_${NAME}.log"
fi

for MODEL_SEED in 0 1 2; do
  CKPT="outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed${MODEL_SEED}/checkpoint.pt"

  apptainer exec --nv "${CONTAINER}" \
    python -u scripts/eval_mixed_hard_tieaware_moving_only.py \
      --data "${DATA_POOL}" \
      --groups "${GROUP_FILE}" \
      --checkpoint "${CKPT}" \
      --out "${ROOT_REP}/moving_only_full_seed${MODEL_SEED}.json" \
      --batch-groups 128 \
      --device cuda \
    2>&1 | tee "${LOG_DIR}/moving_only_full_seed${MODEL_SEED}_${NAME}.log"

  apptainer exec --nv "${CONTAINER}" \
    python -u scripts/eval_state_disjoint_confidence_diagnostics.py \
      --data "${DATA_POOL}" \
      --groups "${GROUP_FILE}" \
      --checkpoint "${CKPT}" \
      --out "${ROOT_REP}/confidence_full_seed${MODEL_SEED}.json" \
      --temperature 0.02 \
      --batch-groups 128 \
      --device cuda \
    2>&1 | tee "${LOG_DIR}/confidence_full_seed${MODEL_SEED}_${NAME}.log"
done

echo "DONE ${NAME}"

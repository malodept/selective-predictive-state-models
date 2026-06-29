#!/usr/bin/env bash
set -euo pipefail

cd /shared/home/mdepastor/projects/spsm

BASE_OUT="outputs/counterfactual/pybullet_obstacles_rgb_v33_ood_grid"
BASE_REP="reports/tables/protocol/pybullet_obstacle_rgb_encoder/v33_ood_grid"
LOG_DIR="logs/v33_ood_grid"

mkdir -p "$BASE_OUT" "$BASE_REP" "$LOG_DIR"

# Format:
# name blocked horizon velocity seed
VARIANTS=(
  "block1_h36_seed20 1 36 1.2 20"
  "block4_h72_seed20 4 72 1.2 20"
)

MODELS=(
  "tiny_full_seed0"
  "small_full_seed0"
  "medium_full_seed0"
  "full_seed0"
)

run_variant () {
  NAME="$1"
  BLOCKED="$2"
  HORIZON="$3"
  VELOCITY="$4"
  SEED="$5"

  OUT_DIR="${BASE_OUT}/${NAME}"
  REP_DIR="${BASE_REP}/${NAME}"

  RAW="${OUT_DIR}/pybullet_obstacle_rgb_v1_ood_${NAME}.npz"
  DINO="${OUT_DIR}/pybullet_obstacle_rgb_v1_ood_${NAME}_dinov2_vits14.npz"
  POOL="${OUT_DIR}/pybullet_obstacle_rgb_v1_ood_${NAME}_dinov2_vits14_pool4.npz"
  GROUPS_FILE="${OUT_DIR}/mixed_hard_state_disjoint_v0_groups.npz"

  mkdir -p "$OUT_DIR" "$REP_DIR"

  echo "======================================================================"
  echo "v33C variant=${NAME} blocked=${BLOCKED} horizon=${HORIZON} velocity=${VELOCITY} seed=${SEED}"
  echo "======================================================================"

  if [ ! -f "$RAW" ]; then
    phipy -u scripts/create_pybullet_obstacle_intervention_rgb_dataset.py \
      --out "$RAW" \
      --report "${REP_DIR}/dataset.json" \
      --groups 2000 \
      --image-size 96 \
      --patch-grid 4 \
      --horizon "$HORIZON" \
      --velocity "$VELOCITY" \
      --num-blocked-directions "$BLOCKED" \
      --seed "$SEED"
  else
    echo "[SKIP] raw exists: $RAW"
  fi

  if [ -f "$POOL" ]; then
    echo "[SKIP] pool exists: $POOL"
  else
    if [ ! -f "$DINO" ]; then
      phipy -u scripts/extract_pybullet_rgb_dinov2_features.py \
        --input "$RAW" \
        --output "$DINO" \
        --report "${REP_DIR}/dinov2_vits14.json" \
        --model dinov2_vits14 \
        --image-size 224 \
        --batch-size 64 \
        --device cuda
    else
      echo "[SKIP] DINO exists: $DINO"
    fi

    phipy -u scripts/pool_dinov2_patchtokens.py \
      --input "$DINO" \
      --output "$POOL" \
      --report "${REP_DIR}/dinov2_vits14_pool4.json" \
      --source-grid 16 \
      --target-grid 4
  fi

  # Keep the compact pooled file; remove huge full-token DINO file to save disk.
  if [ -f "$DINO" ] && [ -f "$POOL" ]; then
    echo "[CLEAN] removing full-token DINO file after pooling: $DINO"
    rm -f "$DINO"
  fi

  if [ ! -f "$GROUPS_FILE" ]; then
    phipy -u scripts/mine_state_disjoint_mixed_hard_groups.py \
      --data "$POOL" \
      --out "$GROUPS_FILE" \
      --report "${REP_DIR}/mixed_hard_state_disjoint_groups.json" \
      --train-groups 8000 \
      --val-groups 1000 \
      --test-groups 1000 \
      --same-state-negatives 2 \
      --same-action-negatives 2 \
      --max-pos-dist 0.15 \
      --prefer-block-mismatch \
      --seed 0
  else
    echo "[SKIP] groups exist: $GROUPS_FILE"
  fi

  for MODEL in "${MODELS[@]}"; do
    CKPT="outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/${MODEL}/checkpoint.pt"

    if [ ! -f "$CKPT" ]; then
      echo "[MISSING CKPT] $CKPT"
      exit 1
    fi

    if [ ! -f "${REP_DIR}/moving_only_${MODEL}.json" ]; then
      phipy -u scripts/eval_mixed_hard_tieaware_moving_only.py \
        --data "$POOL" \
        --groups "$GROUPS_FILE" \
        --checkpoint "$CKPT" \
        --out "${REP_DIR}/moving_only_${MODEL}.json" \
        --batch-groups 128 \
        --device cuda
    else
      echo "[SKIP] moving-only exists: ${REP_DIR}/moving_only_${MODEL}.json"
    fi
  done

  echo "DONE variant ${NAME}"
}

for row in "${VARIANTS[@]}"; do
  run_variant $row
done

echo "DONE v33C dry-run OOD grid"

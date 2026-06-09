#!/usr/bin/env bash
set -euo pipefail

SIF="/shared/projects/phisat2/containers/phisat2.sif"
TRAIN="outputs/tartanair_dinov2_features/features_train.npz"
VAL="outputs/tartanair_dinov2_features/features_val.npz"

RUNS=(
  "cheap5_exp80:5:80"
  "cheap10_exp80:10:80"
  "cheap20_exp80:20:80"
  "cheap10_exp40:10:40"
  "cheap10_exp120:10:120"
)

for SEED in 0 1 2; do
  for SPEC in "${RUNS[@]}"; do
    IFS=: read NAME CHEAP_EPOCHS EXP_EPOCHS <<< "$SPEC"

    echo "=== bestval ${NAME} seed=${SEED} ==="

    apptainer exec --nv "$SIF" \
      python scripts/train_real_selective_refinement_bestval.py \
        --train "$TRAIN" \
        --val "$VAL" \
        --output-dir "outputs/bestval_dinov2_${NAME}_seed${SEED}" \
        --cheap-epochs "$CHEAP_EPOCHS" \
        --expensive-epochs "$EXP_EPOCHS" \
        --reliability-epochs 30 \
        --cheap-hidden 256 \
        --cheap-layers 2 \
        --expensive-hidden 512 \
        --expensive-layers 4 \
        --reliability-hidden 256 \
        --batch-size 256 \
        --lr 0.001 \
        --weight-decay 0.0001 \
        --hard-fraction 0.30 \
        --lambda-compute 0.04 \
        --seed "$SEED" \
        --device cuda
  done
done

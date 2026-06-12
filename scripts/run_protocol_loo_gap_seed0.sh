#!/usr/bin/env bash
set -euo pipefail

SIF="/shared/projects/phisat2/containers/phisat2.sif"
BASE="outputs/tartanair_dinov2_features_v2/loo"

RUNS=(
  "cheap5_exp20:5:20"
  "cheap10_exp20:10:20"
)

for HELDOUT in P000 P001 P002 P003 P004 P005; do
  TRAIN="${BASE}/loo_${HELDOUT}_train.npz"
  VAL="${BASE}/loo_${HELDOUT}_val.npz"

  for SPEC in "${RUNS[@]}"; do
    IFS=: read NAME CHEAP_EPOCHS EXP_EPOCHS <<< "$SPEC"

    echo "=== heldout=${HELDOUT} ${NAME} seed=0 ==="

    apptainer exec --nv "$SIF" \
      python scripts/train_real_selective_refinement_bestval.py \
        --train "$TRAIN" \
        --val "$VAL" \
        --output-dir "outputs/protocol_loo_${HELDOUT}_${NAME}_seed0" \
        --cheap-epochs "$CHEAP_EPOCHS" \
        --expensive-epochs "$EXP_EPOCHS" \
        --reliability-epochs 20 \
        --cheap-hidden 256 \
        --cheap-layers 2 \
        --expensive-hidden 512 \
        --expensive-layers 3 \
        --reliability-hidden 256 \
        --batch-size 256 \
        --lr 0.001 \
        --weight-decay 0.0001 \
        --hard-fraction 0.30 \
        --lambda-compute 0.04 \
        --seed 0 \
        --device cuda
  done
done

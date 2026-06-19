#!/usr/bin/env bash
set -euo pipefail

cd /shared/home/mdepastor/projects/spsm

for SEED in 1 2; do
  for MODE in full action_only state_only no_context; do
    echo "===================================================================================================="
    echo "Mixed hard delta Transformer: mode=${MODE}, seed=${SEED}"
    echo "===================================================================================================="

    apptainer exec --nv /shared/projects/phisat2/containers/phisat2.sif \
      python -u scripts/train_mixed_hard_delta_transformer.py \
        --data outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz \
        --groups outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz \
        --out-dir outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/delta_transformer/${MODE}_seed${SEED} \
        --report reports/tables/protocol/pybullet_obstacle_rgb_encoder/mixed_hard/delta_transformer/${MODE}_seed${SEED}.json \
        --mode ${MODE} \
        --epochs 300 \
        --batch-groups 64 \
        --eval-batch-groups 64 \
        --lr 0.0003 \
        --weight-decay 0.0001 \
        --model-dim 384 \
        --heads 6 \
        --layers 3 \
        --dropout 0.05 \
        --temperature 0.02 \
        --lambda-mse 0.01 \
        --device cuda \
        --seed ${SEED}
  done
done

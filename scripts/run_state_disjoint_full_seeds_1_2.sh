#!/usr/bin/env bash
set -euo pipefail

cd /shared/home/mdepastor/projects/spsm

for SEED in 1 2; do
  echo "===================================================================================================="
  echo "State-disjoint mixed hard delta Transformer: full, seed=${SEED}"
  echo "===================================================================================================="

  apptainer exec --nv /shared/projects/phisat2/containers/phisat2.sif \
    python -u scripts/train_mixed_hard_delta_transformer.py \
      --data outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz \
      --groups outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz \
      --out-dir outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed${SEED} \
      --report reports/tables/protocol/pybullet_obstacle_rgb_encoder/scale_5k/mixed_hard_state_disjoint/delta_transformer/full_seed${SEED}.json \
      --mode full \
      --epochs 100 \
      --batch-groups 64 \
      --eval-batch-groups 128 \
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

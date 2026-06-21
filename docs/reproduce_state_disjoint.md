# Reproducing the SPSM state-disjoint exact-intervention result

This document records the main reproduction chain for the current SPSM exact-intervention benchmark.

## Scientific target

The core result is the state-disjoint mixed-hard PyBullet benchmark using:

- exact simulator interventions,
- RGB observations,
- frozen DINOv2 ViT-S/14 patch tokens,
- 16x16 to 4x4 token pooling,
- mixed hard negatives,
- latent-displacement ranking,
- a spatial action-conditioned delta Transformer,
- tie-aware and moving-only evaluation.

The main reported result is the moving-only strict top-1 accuracy of the full state-action Transformer across seeds 0, 1, and 2.

## Environment

Project root:

```bash
cd /shared/home/mdepastor/projects/spsm

Container used on NEOHPC:

/shared/projects/phisat2/containers/phisat2.sif
Step 1 — Generate PyBullet exact-intervention RGB dataset
apptainer exec --nv /shared/projects/phisat2/containers/phisat2.sif \
  python -u scripts/create_pybullet_obstacle_intervention_rgb_dataset.py \
    --out outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0.npz \
    --report reports/tables/protocol/pybullet_obstacle_rgb_encoder/scale_5k/pybullet_obstacle_rgb_v1_5k_seed0.json \
    --groups 5000 \
    --image-size 96 \
    --patch-grid 4 \
    --horizon 36 \
    --velocity 1.2 \
    --num-blocked-directions 2 \
    --seed 0
Step 2 — Extract DINOv2 ViT-S/14 patch tokens
apptainer exec --nv /shared/projects/phisat2/containers/phisat2.sif \
  python -u scripts/extract_pybullet_rgb_dinov2_features.py \
    --input outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0.npz \
    --output outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14.npz \
    --report reports/tables/protocol/pybullet_obstacle_rgb_encoder/scale_5k/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14.json \
    --model dinov2_vits14 \
    --image-size 224 \
    --batch-size 64 \
    --device cuda
Step 3 — Pool patch tokens from 16x16 to 4x4
apptainer exec --nv /shared/projects/phisat2/containers/phisat2.sif \
  python -u scripts/pool_dinov2_patchtokens.py \
    --input outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14.npz \
    --output outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz \
    --report reports/tables/protocol/pybullet_obstacle_rgb_encoder/scale_5k/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.json \
    --source-grid 16 \
    --target-grid 4
Step 4 — Mine state-disjoint mixed hard-negative groups
apptainer exec --nv /shared/projects/phisat2/containers/phisat2.sif \
  python -u scripts/mine_state_disjoint_mixed_hard_groups.py \
    --data outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz \
    --out outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz \
    --report reports/tables/protocol/pybullet_obstacle_rgb_encoder/scale_5k/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.json \
    --train-groups 8000 \
    --val-groups 1000 \
    --test-groups 1000 \
    --same-state-negatives 2 \
    --same-action-negatives 2 \
    --max-pos-dist 0.15 \
    --prefer-block-mismatch \
    --seed 0
Step 5 — Train full state-action delta Transformer
for SEED in 0 1 2; do
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
Step 6 — Moving-only tie-aware evaluation
for SEED in 0 1 2; do
  apptainer exec --nv /shared/projects/phisat2/containers/phisat2.sif \
    python -u scripts/eval_mixed_hard_tieaware_moving_only.py \
      --data outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz \
      --groups outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz \
      --checkpoint outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed${SEED}/checkpoint.pt \
      --out reports/tables/protocol/pybullet_obstacle_rgb_encoder/scale_5k/mixed_hard_state_disjoint/moving_only_eval/full_seed${SEED}.json \
      --batch-groups 128 \
      --device cuda
done
Step 7 — Confidence diagnostic
for SEED in 0 1 2; do
  apptainer exec --nv /shared/projects/phisat2/containers/phisat2.sif \
    python -u scripts/eval_state_disjoint_confidence_diagnostics.py \
      --data outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz \
      --groups outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz \
      --checkpoint outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed${SEED}/checkpoint.pt \
      --out reports/tables/protocol/pybullet_obstacle_rgb_encoder/scale_5k/mixed_hard_state_disjoint/confidence/full_seed${SEED}.json \
      --temperature 0.02 \
      --batch-groups 128 \
      --device cuda
done
Step 8 — Summarize confidence across seeds
apptainer exec --nv /shared/projects/phisat2/containers/phisat2.sif \
  python -u scripts/summarize_state_disjoint_confidence_multiseed.py


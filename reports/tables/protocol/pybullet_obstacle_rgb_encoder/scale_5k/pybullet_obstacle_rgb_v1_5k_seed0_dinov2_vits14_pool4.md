# DINOv2 patch-token pooling

- input: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- source shape: `(25000, 256, 384)`
- pooled shape: `(25000, 16, 384)`
- source grid: `16x16`
- target grid: `4x4`
- output size GB: `0.3644`

## Interpretation

This keeps frozen DINOv2 dense visual features but reduces the patch-token grid from 16x16 to 4x4 for stable exact-intervention training.

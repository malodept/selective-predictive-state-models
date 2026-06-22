# DINOv2 patch-token pooling

- input: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_h72_seed11/pybullet_obstacle_rgb_v1_ood_block2_h72_seed11_dinov2_vits14.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_h72_seed11/pybullet_obstacle_rgb_v1_ood_block2_h72_seed11_dinov2_vits14_pool4.npz`
- source shape: `(10000, 256, 384)`
- pooled shape: `(10000, 16, 384)`
- source grid: `16x16`
- target grid: `4x4`
- output size GB: `0.1457`

## Interpretation

This keeps frozen DINOv2 dense visual features but reduces the patch-token grid from 16x16 to 4x4 for stable exact-intervention training.

# PyBullet obstacle DINOv2 feature extraction

- input: `outputs/counterfactual/pybullet_obstacles_rgb_v33_ood_grid/block2_h36_seed20/pybullet_obstacle_rgb_v1_ood_block2_h36_seed20.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb_v33_ood_grid/block2_h36_seed20/pybullet_obstacle_rgb_v1_ood_block2_h36_seed20_dinov2_vits14.npz`
- model: `dinov2_vits14`
- image size: `224`
- z_current: `(10000, 256, 384)` `float16`
- z_future: `(10000, 256, 384)` `float16`
- cls_current: `(10000, 384)` `float16`
- cls_future: `(10000, 384)` `float16`
- action: `(10000, 2)`
- candidate groups: `(2000, 5)`
- output size GB: `3.6375`

## Interpretation

This replaces engineered PyBullet patch tokens with frozen DINOv2 patch-token representations extracted from RGB observations.
The resulting file is compatible with the exact-intervention oracle and ablation scripts.

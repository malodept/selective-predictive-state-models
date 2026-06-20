# PyBullet obstacle DINOv2 feature extraction

- input: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14.npz`
- model: `dinov2_vits14`
- image size: `224`
- z_current: `(25000, 256, 384)` `float16`
- z_future: `(25000, 256, 384)` `float16`
- cls_current: `(25000, 384)` `float16`
- cls_future: `(25000, 384)` `float16`
- action: `(25000, 2)`
- candidate groups: `(5000, 5)`
- output size GB: `9.0935`

## Interpretation

This replaces engineered PyBullet patch tokens with frozen DINOv2 patch-token representations extracted from RGB observations.
The resulting file is compatible with the exact-intervention oracle and ablation scripts.

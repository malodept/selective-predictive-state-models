# Mixed hard delta Transformer: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz`
- mode: `full`
- groups count: `10000`
- candidates: `5`
- tokens: `16`
- token dim: `384`
- model dim: `384`
- layers: `3`
- heads: `6`
- train/val/test groups: `8000/1000/1000`
- chance top-1: `0.200000`
- best val top-1: `1.000000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 0.799875 | 0.128638 |
| train | action_zero | 0.289750 | 2.765625 | 0.089625 | -0.060916 |
| train | cond_state_shuffle | 0.558375 | 2.378000 | 0.358250 | -0.039055 |
| train | cond_state_zero | 0.424500 | 2.511500 | 0.305625 | -0.046863 |
| val | original | 1.000000 | 1.000000 | 0.811000 | 0.126588 |
| val | action_zero | 0.273000 | 2.805000 | 0.084000 | -0.063385 |
| val | cond_state_shuffle | 0.559000 | 2.340000 | 0.370000 | -0.032735 |
| val | cond_state_zero | 0.424000 | 2.512000 | 0.306000 | -0.044414 |
| test | original | 0.997000 | 1.005000 | 0.816000 | 0.130948 |
| test | action_zero | 0.276000 | 2.811000 | 0.095000 | -0.062011 |
| test | cond_state_shuffle | 0.554000 | 2.389000 | 0.373000 | -0.043212 |
| test | cond_state_zero | 0.393000 | 2.583000 | 0.295000 | -0.050665 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

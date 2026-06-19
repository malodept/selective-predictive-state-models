# Mixed hard delta Transformer: state_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz`
- mode: `state_only`
- groups count: `1000`
- candidates: `5`
- tokens: `16`
- token dim: `384`
- model dim: `384`
- layers: `3`
- heads: `6`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.350000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.430000 | 2.006250 | 0.430000 | -0.005395 |
| train | action_zero | 0.430000 | 2.006250 | 0.430000 | -0.005395 |
| train | cond_state_shuffle | 0.182500 | 3.093750 | 0.182500 | -0.038595 |
| train | cond_state_zero | 0.111250 | 3.296250 | 0.111250 | -0.056416 |
| val | original | 0.350000 | 2.370000 | 0.350000 | -0.014215 |
| val | action_zero | 0.350000 | 2.370000 | 0.350000 | -0.014215 |
| val | cond_state_shuffle | 0.130000 | 3.260000 | 0.130000 | -0.044276 |
| val | cond_state_zero | 0.130000 | 3.170000 | 0.130000 | -0.051210 |
| test | original | 0.210000 | 2.640000 | 0.210000 | -0.016415 |
| test | action_zero | 0.210000 | 2.640000 | 0.210000 | -0.016415 |
| test | cond_state_shuffle | 0.160000 | 3.400000 | 0.160000 | -0.047516 |
| test | cond_state_zero | 0.170000 | 3.220000 | 0.170000 | -0.056209 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

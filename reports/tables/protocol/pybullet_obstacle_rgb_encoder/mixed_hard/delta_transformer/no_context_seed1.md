# Mixed hard delta Transformer: no_context

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz`
- mode: `no_context`
- groups count: `1000`
- candidates: `5`
- tokens: `16`
- token dim: `384`
- model dim: `384`
- layers: `3`
- heads: `6`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.260000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.296250 | 2.690000 | 0.296250 | -0.013502 |
| train | action_zero | 0.296250 | 2.690000 | 0.296250 | -0.013502 |
| train | cond_state_shuffle | 0.296250 | 2.690000 | 0.296250 | -0.013502 |
| train | cond_state_zero | 0.296250 | 2.690000 | 0.296250 | -0.013502 |
| val | original | 0.260000 | 3.060000 | 0.260000 | -0.020396 |
| val | action_zero | 0.260000 | 3.060000 | 0.260000 | -0.020396 |
| val | cond_state_shuffle | 0.260000 | 3.060000 | 0.260000 | -0.020396 |
| val | cond_state_zero | 0.260000 | 3.060000 | 0.260000 | -0.020396 |
| test | original | 0.200000 | 3.150000 | 0.200000 | -0.024782 |
| test | action_zero | 0.200000 | 3.150000 | 0.200000 | -0.024782 |
| test | cond_state_shuffle | 0.200000 | 3.150000 | 0.200000 | -0.024782 |
| test | cond_state_zero | 0.200000 | 3.150000 | 0.200000 | -0.024782 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

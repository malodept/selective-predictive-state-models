# Mixed hard delta Transformer: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz`
- mode: `full`
- groups count: `1000`
- candidates: `5`
- tokens: `16`
- token dim: `384`
- model dim: `384`
- layers: `3`
- heads: `6`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.960000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 1.000000 | 0.156348 |
| train | action_zero | 0.468750 | 2.022500 | 0.468750 | -0.014849 |
| train | cond_state_shuffle | 0.442500 | 2.475000 | 0.442500 | -0.034243 |
| train | cond_state_zero | 0.395000 | 2.532500 | 0.395000 | -0.032821 |
| val | original | 0.960000 | 1.080000 | 0.960000 | 0.114798 |
| val | action_zero | 0.250000 | 2.610000 | 0.250000 | -0.056911 |
| val | cond_state_shuffle | 0.370000 | 2.650000 | 0.370000 | -0.049771 |
| val | cond_state_zero | 0.350000 | 2.640000 | 0.350000 | -0.046765 |
| test | original | 0.950000 | 1.060000 | 0.950000 | 0.126633 |
| test | action_zero | 0.250000 | 2.510000 | 0.250000 | -0.051042 |
| test | cond_state_shuffle | 0.450000 | 2.300000 | 0.450000 | -0.017382 |
| test | cond_state_zero | 0.390000 | 2.750000 | 0.390000 | -0.040396 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

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
- best val top-1: `0.370000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.428750 | 1.983750 | 0.428750 | -0.004822 |
| train | action_zero | 0.428750 | 1.983750 | 0.428750 | -0.004822 |
| train | cond_state_shuffle | 0.218750 | 3.025000 | 0.218750 | -0.035326 |
| train | cond_state_zero | 0.108750 | 3.310000 | 0.108750 | -0.061608 |
| val | original | 0.370000 | 2.110000 | 0.370000 | -0.012544 |
| val | action_zero | 0.370000 | 2.110000 | 0.370000 | -0.012544 |
| val | cond_state_shuffle | 0.180000 | 3.140000 | 0.180000 | -0.038028 |
| val | cond_state_zero | 0.090000 | 3.310000 | 0.090000 | -0.058398 |
| test | original | 0.290000 | 2.400000 | 0.290000 | -0.017981 |
| test | action_zero | 0.290000 | 2.400000 | 0.290000 | -0.017981 |
| test | cond_state_shuffle | 0.210000 | 2.980000 | 0.210000 | -0.038180 |
| test | cond_state_zero | 0.150000 | 3.210000 | 0.150000 | -0.053070 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

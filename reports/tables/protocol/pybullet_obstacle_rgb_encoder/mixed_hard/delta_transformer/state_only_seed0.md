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
- best val top-1: `0.310000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.463750 | 1.893750 | 0.463750 | -0.002538 |
| train | action_zero | 0.463750 | 1.893750 | 0.463750 | -0.002538 |
| train | cond_state_shuffle | 0.201250 | 2.997500 | 0.201250 | -0.036090 |
| train | cond_state_zero | 0.121250 | 3.267500 | 0.121250 | -0.065404 |
| val | original | 0.310000 | 2.340000 | 0.310000 | -0.015415 |
| val | action_zero | 0.310000 | 2.340000 | 0.310000 | -0.015415 |
| val | cond_state_shuffle | 0.150000 | 3.240000 | 0.150000 | -0.042737 |
| val | cond_state_zero | 0.110000 | 3.480000 | 0.110000 | -0.071559 |
| test | original | 0.250000 | 2.370000 | 0.250000 | -0.016502 |
| test | action_zero | 0.250000 | 2.370000 | 0.250000 | -0.016502 |
| test | cond_state_shuffle | 0.160000 | 3.080000 | 0.160000 | -0.039735 |
| test | cond_state_zero | 0.110000 | 3.410000 | 0.110000 | -0.066352 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

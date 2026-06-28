# Mixed hard delta Transformer: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz`
- mode: `full`
- groups count: `10000`
- candidates: `5`
- tokens: `16`
- token dim: `384`
- model dim: `192`
- layers: `2`
- heads: `4`
- train/val/test groups: `8000/1000/1000`
- chance top-1: `0.200000`
- best val top-1: `1.000000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 0.799875 | 0.126321 |
| train | action_zero | 0.295875 | 2.709875 | 0.095750 | -0.062400 |
| train | cond_state_shuffle | 0.574250 | 2.313375 | 0.374125 | -0.036382 |
| train | cond_state_zero | 0.510000 | 2.416125 | 0.309875 | -0.033343 |
| val | original | 1.000000 | 1.000000 | 0.811000 | 0.123502 |
| val | action_zero | 0.274000 | 2.745000 | 0.085000 | -0.065058 |
| val | cond_state_shuffle | 0.546000 | 2.380000 | 0.357000 | -0.040958 |
| val | cond_state_zero | 0.508000 | 2.416000 | 0.319000 | -0.034497 |
| test | original | 0.998000 | 1.003000 | 0.817000 | 0.128006 |
| test | action_zero | 0.278000 | 2.762000 | 0.097000 | -0.064084 |
| test | cond_state_shuffle | 0.554000 | 2.378000 | 0.373000 | -0.039275 |
| test | cond_state_zero | 0.480000 | 2.503000 | 0.299000 | -0.036453 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

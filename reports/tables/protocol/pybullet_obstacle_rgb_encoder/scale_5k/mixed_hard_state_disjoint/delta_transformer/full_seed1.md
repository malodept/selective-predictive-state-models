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
| train | original | 1.000000 | 1.000000 | 0.799875 | 0.127714 |
| train | action_zero | 0.294625 | 2.751375 | 0.094500 | -0.063637 |
| train | cond_state_shuffle | 0.560250 | 2.383750 | 0.360125 | -0.041252 |
| train | cond_state_zero | 0.482125 | 2.534875 | 0.346000 | -0.052728 |
| val | original | 1.000000 | 1.000000 | 0.811000 | 0.125049 |
| val | action_zero | 0.280000 | 2.808000 | 0.091000 | -0.065882 |
| val | cond_state_shuffle | 0.586000 | 2.275000 | 0.397000 | -0.030273 |
| val | cond_state_zero | 0.488000 | 2.548000 | 0.357000 | -0.052632 |
| test | original | 0.998000 | 1.002000 | 0.817000 | 0.128081 |
| test | action_zero | 0.276000 | 2.815000 | 0.095000 | -0.064503 |
| test | cond_state_shuffle | 0.537000 | 2.473000 | 0.356000 | -0.049949 |
| test | cond_state_zero | 0.473000 | 2.624000 | 0.351000 | -0.059245 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

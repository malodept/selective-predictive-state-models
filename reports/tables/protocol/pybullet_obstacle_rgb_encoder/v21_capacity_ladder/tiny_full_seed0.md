# Mixed hard delta Transformer: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz`
- mode: `full`
- groups count: `10000`
- candidates: `5`
- tokens: `16`
- token dim: `384`
- model dim: `64`
- layers: `1`
- heads: `4`
- train/val/test groups: `8000/1000/1000`
- chance top-1: `0.200000`
- best val top-1: `1.000000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 0.799875 | 0.123508 |
| train | action_zero | 0.299125 | 2.707750 | 0.099000 | -0.064568 |
| train | cond_state_shuffle | 0.562125 | 2.291750 | 0.362000 | -0.037542 |
| train | cond_state_zero | 0.512250 | 2.490750 | 0.313500 | -0.049152 |
| val | original | 1.000000 | 1.000000 | 0.811000 | 0.121403 |
| val | action_zero | 0.284000 | 2.735000 | 0.095000 | -0.066752 |
| val | cond_state_shuffle | 0.543000 | 2.347000 | 0.354000 | -0.037906 |
| val | cond_state_zero | 0.508000 | 2.514000 | 0.321000 | -0.048428 |
| test | original | 0.998000 | 1.002000 | 0.817000 | 0.124866 |
| test | action_zero | 0.293000 | 2.743000 | 0.112000 | -0.066040 |
| test | cond_state_shuffle | 0.545000 | 2.372000 | 0.364000 | -0.041602 |
| test | cond_state_zero | 0.502000 | 2.629000 | 0.321000 | -0.054601 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

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
- best val top-1: `0.320000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.255000 | 2.861250 | 0.255000 | -0.018000 |
| train | action_zero | 0.255000 | 2.861250 | 0.255000 | -0.018000 |
| train | cond_state_shuffle | 0.255000 | 2.861250 | 0.255000 | -0.018000 |
| train | cond_state_zero | 0.255000 | 2.861250 | 0.255000 | -0.018000 |
| val | original | 0.320000 | 2.670000 | 0.320000 | -0.018060 |
| val | action_zero | 0.320000 | 2.670000 | 0.320000 | -0.018060 |
| val | cond_state_shuffle | 0.320000 | 2.670000 | 0.320000 | -0.018060 |
| val | cond_state_zero | 0.320000 | 2.670000 | 0.320000 | -0.018060 |
| test | original | 0.160000 | 3.350000 | 0.160000 | -0.026482 |
| test | action_zero | 0.160000 | 3.350000 | 0.160000 | -0.026482 |
| test | cond_state_shuffle | 0.160000 | 3.350000 | 0.160000 | -0.026482 |
| test | cond_state_zero | 0.160000 | 3.350000 | 0.160000 | -0.026482 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

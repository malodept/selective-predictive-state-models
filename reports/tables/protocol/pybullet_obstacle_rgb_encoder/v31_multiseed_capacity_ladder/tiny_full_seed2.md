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
| train | original | 1.000000 | 1.000000 | 0.799875 | 0.123886 |
| train | action_zero | 0.302000 | 2.668875 | 0.101875 | -0.061845 |
| train | cond_state_shuffle | 0.571750 | 2.289500 | 0.371625 | -0.036217 |
| train | cond_state_zero | 0.379000 | 2.541125 | 0.293750 | -0.034905 |
| val | original | 1.000000 | 1.000000 | 0.811000 | 0.121664 |
| val | action_zero | 0.289000 | 2.726000 | 0.100000 | -0.064011 |
| val | cond_state_shuffle | 0.554000 | 2.328000 | 0.365000 | -0.041354 |
| val | cond_state_zero | 0.409000 | 2.496000 | 0.313000 | -0.032566 |
| test | original | 0.999000 | 1.003000 | 0.818000 | 0.124023 |
| test | action_zero | 0.292000 | 2.718000 | 0.111000 | -0.063582 |
| test | cond_state_shuffle | 0.576000 | 2.273000 | 0.395000 | -0.036835 |
| test | cond_state_zero | 0.339000 | 2.680000 | 0.274000 | -0.040552 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

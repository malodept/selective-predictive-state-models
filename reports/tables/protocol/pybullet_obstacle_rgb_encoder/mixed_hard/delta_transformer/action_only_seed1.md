# Mixed hard delta Transformer: action_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz`
- mode: `action_only`
- groups count: `1000`
- candidates: `5`
- tokens: `16`
- token dim: `384`
- model dim: `384`
- layers: `3`
- heads: `6`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.410000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.536250 | 1.766250 | 0.536250 | 0.002303 |
| train | action_zero | 0.255000 | 2.832500 | 0.255000 | -0.028309 |
| train | cond_state_shuffle | 0.536250 | 1.766250 | 0.536250 | 0.002303 |
| train | cond_state_zero | 0.536250 | 1.766250 | 0.536250 | 0.002303 |
| val | original | 0.410000 | 2.090000 | 0.410000 | -0.014366 |
| val | action_zero | 0.230000 | 2.890000 | 0.230000 | -0.029481 |
| val | cond_state_shuffle | 0.410000 | 2.090000 | 0.410000 | -0.014366 |
| val | cond_state_zero | 0.410000 | 2.090000 | 0.410000 | -0.014366 |
| test | original | 0.250000 | 2.360000 | 0.250000 | -0.020683 |
| test | action_zero | 0.120000 | 3.300000 | 0.120000 | -0.038752 |
| test | cond_state_shuffle | 0.250000 | 2.360000 | 0.250000 | -0.020683 |
| test | cond_state_zero | 0.250000 | 2.360000 | 0.250000 | -0.020683 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

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
- best val top-1: `0.980000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 1.000000 | 0.159979 |
| train | action_zero | 0.436250 | 2.206250 | 0.436250 | -0.015867 |
| train | cond_state_shuffle | 0.418750 | 2.703750 | 0.418750 | -0.041502 |
| train | cond_state_zero | 0.406250 | 2.708750 | 0.406250 | -0.047572 |
| val | original | 0.980000 | 1.040000 | 0.980000 | 0.114476 |
| val | action_zero | 0.330000 | 2.600000 | 0.330000 | -0.037271 |
| val | cond_state_shuffle | 0.330000 | 2.900000 | 0.330000 | -0.062367 |
| val | cond_state_zero | 0.420000 | 2.710000 | 0.420000 | -0.067233 |
| test | original | 0.970000 | 1.070000 | 0.970000 | 0.123533 |
| test | action_zero | 0.200000 | 3.000000 | 0.200000 | -0.049262 |
| test | cond_state_shuffle | 0.330000 | 3.120000 | 0.330000 | -0.062268 |
| test | cond_state_zero | 0.320000 | 2.890000 | 0.320000 | -0.073278 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

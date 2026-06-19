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
- best val top-1: `0.950000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 1.000000 | 0.144014 |
| train | action_zero | 0.460000 | 2.101250 | 0.460000 | -0.012897 |
| train | cond_state_shuffle | 0.421250 | 2.550000 | 0.421250 | -0.037678 |
| train | cond_state_zero | 0.286250 | 2.650000 | 0.286250 | -0.038125 |
| val | original | 0.950000 | 1.090000 | 0.950000 | 0.095948 |
| val | action_zero | 0.350000 | 2.410000 | 0.350000 | -0.029825 |
| val | cond_state_shuffle | 0.410000 | 2.710000 | 0.410000 | -0.039618 |
| val | cond_state_zero | 0.280000 | 2.640000 | 0.280000 | -0.034403 |
| test | original | 0.940000 | 1.100000 | 0.940000 | 0.095756 |
| test | action_zero | 0.340000 | 2.480000 | 0.340000 | -0.035575 |
| test | cond_state_shuffle | 0.450000 | 2.360000 | 0.450000 | -0.017232 |
| test | cond_state_zero | 0.280000 | 2.700000 | 0.280000 | -0.042506 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

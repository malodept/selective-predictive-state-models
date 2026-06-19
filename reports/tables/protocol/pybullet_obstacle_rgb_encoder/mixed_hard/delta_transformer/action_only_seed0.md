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
- best val top-1: `0.310000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.382500 | 2.227500 | 0.382500 | -0.009232 |
| train | action_zero | 0.228750 | 2.926250 | 0.228750 | -0.025816 |
| train | cond_state_shuffle | 0.382500 | 2.227500 | 0.382500 | -0.009232 |
| train | cond_state_zero | 0.382500 | 2.227500 | 0.382500 | -0.009232 |
| val | original | 0.310000 | 2.640000 | 0.310000 | -0.019503 |
| val | action_zero | 0.210000 | 3.080000 | 0.210000 | -0.029974 |
| val | cond_state_shuffle | 0.310000 | 2.640000 | 0.310000 | -0.019503 |
| val | cond_state_zero | 0.310000 | 2.640000 | 0.310000 | -0.019503 |
| test | original | 0.280000 | 2.460000 | 0.280000 | -0.016463 |
| test | action_zero | 0.280000 | 2.790000 | 0.280000 | -0.024152 |
| test | cond_state_shuffle | 0.280000 | 2.460000 | 0.280000 | -0.016463 |
| test | cond_state_zero | 0.280000 | 2.460000 | 0.280000 | -0.016463 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

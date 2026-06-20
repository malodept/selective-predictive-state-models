# Mixed hard delta Transformer: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard/mixed_hard_v0_10k_groups.npz`
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
| train | original | 0.999375 | 1.001125 | 0.999375 | 0.162224 |
| train | action_zero | 0.371250 | 2.488625 | 0.371250 | -0.041082 |
| train | cond_state_shuffle | 0.456750 | 2.668250 | 0.456750 | -0.042722 |
| train | cond_state_zero | 0.417375 | 2.849500 | 0.417375 | -0.060296 |
| val | original | 1.000000 | 1.000000 | 1.000000 | 0.159565 |
| val | action_zero | 0.381000 | 2.478000 | 0.381000 | -0.044708 |
| val | cond_state_shuffle | 0.452000 | 2.745000 | 0.452000 | -0.044946 |
| val | cond_state_zero | 0.444000 | 2.742000 | 0.444000 | -0.053019 |
| test | original | 0.998000 | 1.006000 | 0.998000 | 0.159213 |
| test | action_zero | 0.382000 | 2.455000 | 0.382000 | -0.041052 |
| test | cond_state_shuffle | 0.459000 | 2.652000 | 0.459000 | -0.042565 |
| test | cond_state_zero | 0.426000 | 2.817000 | 0.426000 | -0.060319 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

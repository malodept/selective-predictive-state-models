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
| train | original | 0.999875 | 1.000125 | 0.799750 | 0.119494 |
| train | action_zero | 0.300125 | 2.676375 | 0.100000 | -0.061908 |
| train | cond_state_shuffle | 0.572000 | 2.274250 | 0.371875 | -0.035335 |
| train | cond_state_zero | 0.553500 | 2.449625 | 0.353375 | -0.041245 |
| val | original | 1.000000 | 1.000000 | 0.811000 | 0.117540 |
| val | action_zero | 0.281000 | 2.733000 | 0.092000 | -0.064381 |
| val | cond_state_shuffle | 0.556000 | 2.324000 | 0.367000 | -0.040659 |
| val | cond_state_zero | 0.551000 | 2.459000 | 0.362000 | -0.040667 |
| test | original | 0.998000 | 1.005000 | 0.817000 | 0.122354 |
| test | action_zero | 0.285000 | 2.736000 | 0.104000 | -0.063413 |
| test | cond_state_shuffle | 0.577000 | 2.275000 | 0.396000 | -0.037369 |
| test | cond_state_zero | 0.527000 | 2.565000 | 0.346000 | -0.046301 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

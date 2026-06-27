# Mixed hard delta Transformer: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz`
- mode: `full`
- groups count: `10000`
- candidates: `5`
- tokens: `16`
- token dim: `384`
- model dim: `128`
- layers: `1`
- heads: `4`
- train/val/test groups: `8000/1000/1000`
- chance top-1: `0.200000`
- best val top-1: `1.000000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 0.799875 | 0.125615 |
| train | action_zero | 0.299000 | 2.711000 | 0.098875 | -0.063144 |
| train | cond_state_shuffle | 0.571375 | 2.310375 | 0.371250 | -0.036521 |
| train | cond_state_zero | 0.413750 | 2.500375 | 0.272750 | -0.038985 |
| val | original | 1.000000 | 1.000000 | 0.811000 | 0.124047 |
| val | action_zero | 0.287000 | 2.755000 | 0.098000 | -0.066025 |
| val | cond_state_shuffle | 0.554000 | 2.391000 | 0.365000 | -0.041615 |
| val | cond_state_zero | 0.403000 | 2.540000 | 0.283000 | -0.039365 |
| test | original | 0.996000 | 1.005000 | 0.815000 | 0.125808 |
| test | action_zero | 0.281000 | 2.789000 | 0.100000 | -0.065172 |
| test | cond_state_shuffle | 0.553000 | 2.363000 | 0.372000 | -0.041353 |
| test | cond_state_zero | 0.364000 | 2.612000 | 0.247000 | -0.048793 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

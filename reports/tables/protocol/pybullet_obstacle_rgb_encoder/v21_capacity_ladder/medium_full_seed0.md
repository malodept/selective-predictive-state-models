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
| train | original | 0.999625 | 1.000375 | 0.799500 | 0.121854 |
| train | action_zero | 0.298500 | 2.712375 | 0.098375 | -0.062719 |
| train | cond_state_shuffle | 0.553250 | 2.349875 | 0.353125 | -0.041223 |
| train | cond_state_zero | 0.495625 | 2.420250 | 0.310250 | -0.027968 |
| val | original | 1.000000 | 1.000000 | 0.811000 | 0.119787 |
| val | action_zero | 0.284000 | 2.754000 | 0.095000 | -0.065648 |
| val | cond_state_shuffle | 0.544000 | 2.388000 | 0.355000 | -0.041403 |
| val | cond_state_zero | 0.496000 | 2.417000 | 0.332000 | -0.026966 |
| test | original | 0.997000 | 1.003000 | 0.816000 | 0.123764 |
| test | action_zero | 0.280000 | 2.784000 | 0.099000 | -0.064375 |
| test | cond_state_shuffle | 0.540000 | 2.412000 | 0.359000 | -0.045909 |
| test | cond_state_zero | 0.459000 | 2.511000 | 0.292000 | -0.031883 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

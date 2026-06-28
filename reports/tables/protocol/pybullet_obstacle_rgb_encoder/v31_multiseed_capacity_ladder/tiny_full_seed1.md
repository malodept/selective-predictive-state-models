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
| train | original | 1.000000 | 1.000000 | 0.799875 | 0.124934 |
| train | action_zero | 0.301375 | 2.662875 | 0.101250 | -0.062775 |
| train | cond_state_shuffle | 0.579000 | 2.269500 | 0.379000 | -0.035238 |
| train | cond_state_zero | 0.526625 | 2.521000 | 0.361375 | -0.048577 |
| val | original | 1.000000 | 1.000000 | 0.811000 | 0.124038 |
| val | action_zero | 0.290000 | 2.715000 | 0.101000 | -0.064716 |
| val | cond_state_shuffle | 0.558000 | 2.335000 | 0.369000 | -0.041192 |
| val | cond_state_zero | 0.520000 | 2.535000 | 0.365000 | -0.047155 |
| test | original | 0.996000 | 1.006000 | 0.815000 | 0.126828 |
| test | action_zero | 0.285000 | 2.734000 | 0.104000 | -0.064990 |
| test | cond_state_shuffle | 0.553000 | 2.355000 | 0.372000 | -0.041065 |
| test | cond_state_zero | 0.492000 | 2.649000 | 0.343000 | -0.058752 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

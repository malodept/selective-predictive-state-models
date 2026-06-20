# Mixed hard delta Transformer: no_context

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard/mixed_hard_v0_10k_groups.npz`
- mode: `no_context`
- groups count: `10000`
- candidates: `5`
- tokens: `16`
- token dim: `384`
- model dim: `384`
- layers: `3`
- heads: `6`
- train/val/test groups: `8000/1000/1000`
- chance top-1: `0.200000`
- best val top-1: `0.210000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.231500 | 2.912750 | 0.231250 | -0.018268 |
| train | action_zero | 0.231500 | 2.912750 | 0.231250 | -0.018268 |
| train | cond_state_shuffle | 0.231500 | 2.912750 | 0.231250 | -0.018268 |
| train | cond_state_zero | 0.231500 | 2.912750 | 0.231250 | -0.018268 |
| val | original | 0.210000 | 3.033000 | 0.210000 | -0.021437 |
| val | action_zero | 0.210000 | 3.033000 | 0.210000 | -0.021437 |
| val | cond_state_shuffle | 0.210000 | 3.033000 | 0.210000 | -0.021437 |
| val | cond_state_zero | 0.210000 | 3.033000 | 0.210000 | -0.021437 |
| test | original | 0.222000 | 2.918000 | 0.222000 | -0.019146 |
| test | action_zero | 0.222000 | 2.918000 | 0.222000 | -0.019146 |
| test | cond_state_shuffle | 0.222000 | 2.918000 | 0.222000 | -0.019146 |
| test | cond_state_zero | 0.222000 | 2.918000 | 0.222000 | -0.019146 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

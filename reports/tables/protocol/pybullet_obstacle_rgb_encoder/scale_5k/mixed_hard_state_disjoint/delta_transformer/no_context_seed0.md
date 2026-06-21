# Mixed hard delta Transformer: no_context

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz`
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
- best val top-1: `0.265000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.275375 | 2.727125 | 0.142125 | -0.015963 |
| train | action_zero | 0.275375 | 2.727125 | 0.142125 | -0.015963 |
| train | cond_state_shuffle | 0.275375 | 2.727125 | 0.142125 | -0.015963 |
| train | cond_state_zero | 0.275375 | 2.727125 | 0.142125 | -0.015963 |
| val | original | 0.265000 | 2.817000 | 0.142000 | -0.018293 |
| val | action_zero | 0.265000 | 2.817000 | 0.142000 | -0.018293 |
| val | cond_state_shuffle | 0.265000 | 2.817000 | 0.142000 | -0.018293 |
| val | cond_state_zero | 0.265000 | 2.817000 | 0.142000 | -0.018293 |
| test | original | 0.263000 | 2.770000 | 0.157000 | -0.018065 |
| test | action_zero | 0.263000 | 2.770000 | 0.157000 | -0.018065 |
| test | cond_state_shuffle | 0.263000 | 2.770000 | 0.157000 | -0.018065 |
| test | cond_state_zero | 0.263000 | 2.770000 | 0.157000 | -0.018065 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

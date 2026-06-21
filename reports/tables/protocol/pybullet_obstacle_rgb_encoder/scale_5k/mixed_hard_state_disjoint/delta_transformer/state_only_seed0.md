# Mixed hard delta Transformer: state_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz`
- mode: `state_only`
- groups count: `10000`
- candidates: `5`
- tokens: `16`
- token dim: `384`
- model dim: `384`
- layers: `3`
- heads: `6`
- train/val/test groups: `8000/1000/1000`
- chance top-1: `0.200000`
- best val top-1: `0.333000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.753375 | 1.293250 | 0.637250 | 0.018957 |
| train | action_zero | 0.753375 | 1.293250 | 0.637250 | 0.018957 |
| train | cond_state_shuffle | 0.295125 | 2.758000 | 0.114000 | -0.046609 |
| train | cond_state_zero | 0.297375 | 2.758625 | 0.097750 | -0.045394 |
| val | original | 0.333000 | 2.122000 | 0.271000 | -0.016573 |
| val | action_zero | 0.333000 | 2.122000 | 0.271000 | -0.016573 |
| val | cond_state_shuffle | 0.270000 | 2.812000 | 0.096000 | -0.050241 |
| val | cond_state_zero | 0.270000 | 2.831000 | 0.082000 | -0.047317 |
| test | original | 0.339000 | 2.139000 | 0.281000 | -0.016054 |
| test | action_zero | 0.339000 | 2.139000 | 0.281000 | -0.016054 |
| test | cond_state_shuffle | 0.265000 | 2.850000 | 0.113000 | -0.048789 |
| test | cond_state_zero | 0.275000 | 2.816000 | 0.097000 | -0.045430 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

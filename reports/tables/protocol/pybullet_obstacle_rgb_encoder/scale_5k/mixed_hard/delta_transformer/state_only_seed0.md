# Mixed hard delta Transformer: state_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard/mixed_hard_v0_10k_groups.npz`
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
- best val top-1: `0.373000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.474875 | 1.726500 | 0.474875 | -0.001608 |
| train | action_zero | 0.474875 | 1.726500 | 0.474875 | -0.001608 |
| train | cond_state_shuffle | 0.182500 | 3.112000 | 0.182500 | -0.046524 |
| train | cond_state_zero | 0.132625 | 3.243500 | 0.132625 | -0.067240 |
| val | original | 0.373000 | 1.949000 | 0.373000 | -0.008455 |
| val | action_zero | 0.373000 | 1.949000 | 0.373000 | -0.008455 |
| val | cond_state_shuffle | 0.184000 | 3.122000 | 0.184000 | -0.049017 |
| val | cond_state_zero | 0.137000 | 3.187000 | 0.137000 | -0.066938 |
| test | original | 0.358000 | 1.983000 | 0.358000 | -0.009890 |
| test | action_zero | 0.358000 | 1.983000 | 0.358000 | -0.009890 |
| test | cond_state_shuffle | 0.172000 | 3.126000 | 0.172000 | -0.046308 |
| test | cond_state_zero | 0.151000 | 3.228000 | 0.151000 | -0.066547 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

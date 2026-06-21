# Mixed hard delta Transformer: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz`
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
| train | original | 1.000000 | 1.000000 | 0.799875 | 0.129942 |
| train | action_zero | 0.289000 | 2.786500 | 0.088875 | -0.064395 |
| train | cond_state_shuffle | 0.548750 | 2.392625 | 0.348625 | -0.040377 |
| train | cond_state_zero | 0.479875 | 2.554000 | 0.335500 | -0.055448 |
| val | original | 1.000000 | 1.000000 | 0.811000 | 0.129017 |
| val | action_zero | 0.271000 | 2.855000 | 0.082000 | -0.067124 |
| val | cond_state_shuffle | 0.547000 | 2.401000 | 0.358000 | -0.039180 |
| val | cond_state_zero | 0.484000 | 2.574000 | 0.341000 | -0.057430 |
| test | original | 0.999000 | 1.001000 | 0.818000 | 0.130859 |
| test | action_zero | 0.277000 | 2.852000 | 0.096000 | -0.065557 |
| test | cond_state_shuffle | 0.510000 | 2.516000 | 0.329000 | -0.047576 |
| test | cond_state_zero | 0.449000 | 2.681000 | 0.331000 | -0.064467 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

# Mixed hard delta Transformer: action_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz`
- mode: `action_only`
- groups count: `10000`
- candidates: `5`
- tokens: `16`
- token dim: `384`
- model dim: `384`
- layers: `3`
- heads: `6`
- train/val/test groups: `8000/1000/1000`
- chance top-1: `0.200000`
- best val top-1: `0.486000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.519000 | 1.822125 | 0.318875 | -0.006993 |
| train | action_zero | 0.296250 | 2.772375 | 0.096125 | -0.062784 |
| train | cond_state_shuffle | 0.519000 | 1.822125 | 0.318875 | -0.006993 |
| train | cond_state_zero | 0.519000 | 1.822125 | 0.318875 | -0.006993 |
| val | original | 0.486000 | 1.910000 | 0.297000 | -0.011059 |
| val | action_zero | 0.291000 | 2.829000 | 0.102000 | -0.064404 |
| val | cond_state_shuffle | 0.486000 | 1.910000 | 0.297000 | -0.011059 |
| val | cond_state_zero | 0.486000 | 1.910000 | 0.297000 | -0.011059 |
| test | original | 0.449000 | 1.985000 | 0.268000 | -0.012572 |
| test | action_zero | 0.285000 | 2.838000 | 0.104000 | -0.063621 |
| test | cond_state_shuffle | 0.449000 | 1.985000 | 0.268000 | -0.012572 |
| test | cond_state_zero | 0.449000 | 1.985000 | 0.268000 | -0.012572 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

# Mixed hard delta Transformer: action_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz`
- mode: `action_only`
- groups count: `1000`
- candidates: `5`
- tokens: `16`
- token dim: `384`
- model dim: `384`
- layers: `3`
- heads: `6`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.400000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.468750 | 1.937500 | 0.468750 | -0.002884 |
| train | action_zero | 0.231250 | 2.843750 | 0.231250 | -0.026584 |
| train | cond_state_shuffle | 0.468750 | 1.937500 | 0.468750 | -0.002884 |
| train | cond_state_zero | 0.468750 | 1.937500 | 0.468750 | -0.002884 |
| val | original | 0.400000 | 2.240000 | 0.400000 | -0.016296 |
| val | action_zero | 0.230000 | 2.940000 | 0.230000 | -0.028211 |
| val | cond_state_shuffle | 0.400000 | 2.240000 | 0.400000 | -0.016296 |
| val | cond_state_zero | 0.400000 | 2.240000 | 0.400000 | -0.016296 |
| test | original | 0.350000 | 2.450000 | 0.350000 | -0.018884 |
| test | action_zero | 0.180000 | 3.210000 | 0.180000 | -0.037922 |
| test | cond_state_shuffle | 0.350000 | 2.450000 | 0.350000 | -0.018884 |
| test | cond_state_zero | 0.350000 | 2.450000 | 0.350000 | -0.018884 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

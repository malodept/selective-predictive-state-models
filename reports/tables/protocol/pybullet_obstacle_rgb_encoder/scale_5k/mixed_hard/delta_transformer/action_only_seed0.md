# Mixed hard delta Transformer: action_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard/mixed_hard_v0_10k_groups.npz`
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
- best val top-1: `0.369000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.410625 | 2.030625 | 0.410625 | -0.007291 |
| train | action_zero | 0.174750 | 3.062125 | 0.174750 | -0.037783 |
| train | cond_state_shuffle | 0.410625 | 2.030625 | 0.410625 | -0.007291 |
| train | cond_state_zero | 0.410625 | 2.030625 | 0.410625 | -0.007291 |
| val | original | 0.369000 | 2.168000 | 0.369000 | -0.012855 |
| val | action_zero | 0.157000 | 3.112000 | 0.157000 | -0.039831 |
| val | cond_state_shuffle | 0.369000 | 2.168000 | 0.369000 | -0.012855 |
| val | cond_state_zero | 0.369000 | 2.168000 | 0.369000 | -0.012855 |
| test | original | 0.362000 | 2.129000 | 0.362000 | -0.012394 |
| test | action_zero | 0.176000 | 3.092000 | 0.176000 | -0.038735 |
| test | cond_state_shuffle | 0.362000 | 2.129000 | 0.362000 | -0.012394 |
| test | cond_state_zero | 0.362000 | 2.129000 | 0.362000 | -0.012394 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

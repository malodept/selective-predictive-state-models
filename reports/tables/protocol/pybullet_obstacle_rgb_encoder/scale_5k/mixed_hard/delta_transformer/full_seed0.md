# Mixed hard delta Transformer: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard/mixed_hard_v0_10k_groups.npz`
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
| train | original | 1.000000 | 1.000000 | 1.000000 | 0.160479 |
| train | action_zero | 0.350125 | 2.504250 | 0.350125 | -0.031679 |
| train | cond_state_shuffle | 0.464250 | 2.671125 | 0.464250 | -0.043159 |
| train | cond_state_zero | 0.426375 | 2.845875 | 0.426375 | -0.071725 |
| val | original | 1.000000 | 1.000000 | 1.000000 | 0.153743 |
| val | action_zero | 0.343000 | 2.590000 | 0.343000 | -0.035430 |
| val | cond_state_shuffle | 0.455000 | 2.746000 | 0.455000 | -0.051991 |
| val | cond_state_zero | 0.444000 | 2.763000 | 0.444000 | -0.068978 |
| test | original | 0.998000 | 1.003000 | 0.998000 | 0.156647 |
| test | action_zero | 0.360000 | 2.535000 | 0.360000 | -0.033520 |
| test | cond_state_shuffle | 0.472000 | 2.651000 | 0.472000 | -0.041701 |
| test | cond_state_zero | 0.405000 | 2.931000 | 0.405000 | -0.076716 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

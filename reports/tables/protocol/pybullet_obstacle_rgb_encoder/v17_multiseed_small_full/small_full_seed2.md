# Mixed hard delta Transformer: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz`
- mode: `full`
- groups count: `10000`
- candidates: `5`
- tokens: `16`
- token dim: `384`
- model dim: `128`
- layers: `1`
- heads: `4`
- train/val/test groups: `8000/1000/1000`
- chance top-1: `0.200000`
- best val top-1: `0.999000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 0.799875 | 0.126641 |
| train | action_zero | 0.297000 | 2.723000 | 0.096875 | -0.065157 |
| train | cond_state_shuffle | 0.579875 | 2.283625 | 0.379750 | -0.034715 |
| train | cond_state_zero | 0.485625 | 2.443625 | 0.303125 | -0.040651 |
| val | original | 0.999000 | 1.001000 | 0.810000 | 0.122731 |
| val | action_zero | 0.279000 | 2.812000 | 0.090000 | -0.067970 |
| val | cond_state_shuffle | 0.570000 | 2.316000 | 0.381000 | -0.039725 |
| val | cond_state_zero | 0.452000 | 2.542000 | 0.281000 | -0.046363 |
| test | original | 0.998000 | 1.003000 | 0.817000 | 0.126849 |
| test | action_zero | 0.284000 | 2.796000 | 0.103000 | -0.066492 |
| test | cond_state_shuffle | 0.584000 | 2.256000 | 0.403000 | -0.034336 |
| test | cond_state_zero | 0.467000 | 2.532000 | 0.301000 | -0.047184 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

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
| train | original | 1.000000 | 1.000000 | 1.000000 | 0.168586 |
| train | action_zero | 0.358125 | 2.509500 | 0.358125 | -0.040724 |
| train | cond_state_shuffle | 0.458000 | 2.692375 | 0.458000 | -0.043051 |
| train | cond_state_zero | 0.428000 | 2.781625 | 0.428000 | -0.052879 |
| val | original | 1.000000 | 1.000000 | 1.000000 | 0.168723 |
| val | action_zero | 0.367000 | 2.533000 | 0.367000 | -0.040964 |
| val | cond_state_shuffle | 0.469000 | 2.657000 | 0.469000 | -0.036326 |
| val | cond_state_zero | 0.431000 | 2.811000 | 0.431000 | -0.054622 |
| test | original | 0.997000 | 1.005000 | 0.997000 | 0.163988 |
| test | action_zero | 0.361000 | 2.457000 | 0.361000 | -0.040504 |
| test | cond_state_shuffle | 0.447000 | 2.765000 | 0.447000 | -0.046821 |
| test | cond_state_zero | 0.431000 | 2.778000 | 0.431000 | -0.054059 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

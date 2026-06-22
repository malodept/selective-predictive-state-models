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
- best val top-1: `0.998000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 0.799875 | 0.122714 |
| train | action_zero | 0.308000 | 2.657125 | 0.107875 | -0.063558 |
| train | cond_state_shuffle | 0.561375 | 2.213750 | 0.361250 | -0.034908 |
| train | cond_state_zero | 0.451125 | 2.512375 | 0.312875 | -0.057838 |
| val | original | 0.998000 | 1.003000 | 0.809000 | 0.120691 |
| val | action_zero | 0.289000 | 2.695000 | 0.100000 | -0.066425 |
| val | cond_state_shuffle | 0.558000 | 2.206000 | 0.369000 | -0.035985 |
| val | cond_state_zero | 0.448000 | 2.508000 | 0.307000 | -0.056341 |
| test | original | 0.999000 | 1.002000 | 0.818000 | 0.125038 |
| test | action_zero | 0.291000 | 2.720000 | 0.110000 | -0.065430 |
| test | cond_state_shuffle | 0.531000 | 2.329000 | 0.350000 | -0.040156 |
| test | cond_state_zero | 0.427000 | 2.600000 | 0.312000 | -0.063435 |

## Interpretation

This model predicts latent displacement tokens using a spatial Transformer over DINOv2 patch tokens.
It is evaluated on mixed hard negatives in delta space.

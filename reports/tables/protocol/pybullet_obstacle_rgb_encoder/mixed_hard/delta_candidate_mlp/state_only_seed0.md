# Same-action delta-candidate MLP: state_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz`
- mode: `state_only`
- groups count: `1000`
- candidates: `5`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.250000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.316250 | 2.517500 | 0.316250 | -0.015006 |
| train | action_zero | 0.316250 | 2.517500 | 0.316250 | -0.015006 |
| train | cond_state_shuffle | 0.293750 | 2.612500 | 0.293750 | -0.018021 |
| train | cond_state_zero | 0.128750 | 3.261250 | 0.128750 | -0.051440 |
| val | original | 0.250000 | 2.950000 | 0.250000 | -0.027617 |
| val | action_zero | 0.250000 | 2.950000 | 0.250000 | -0.027617 |
| val | cond_state_shuffle | 0.210000 | 3.110000 | 0.210000 | -0.030748 |
| val | cond_state_zero | 0.090000 | 3.380000 | 0.090000 | -0.056132 |
| test | original | 0.250000 | 2.830000 | 0.250000 | -0.026595 |
| test | action_zero | 0.250000 | 2.830000 | 0.250000 | -0.026595 |
| test | cond_state_shuffle | 0.250000 | 2.910000 | 0.250000 | -0.028312 |
| test | cond_state_zero | 0.140000 | 3.400000 | 0.140000 | -0.053260 |

## Interpretation

This ranks predicted latent displacements against candidate latent displacements, removing the absolute-state shortcut from future-space ranking.

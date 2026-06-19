# Same-action delta-candidate MLP: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/same_action_hard/same_action_hard_v0_groups.npz`
- mode: `full`
- groups count: `1000`
- candidates: `5`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.900000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 1.000000 | 0.151680 |
| train | action_zero | 1.000000 | 1.000000 | 1.000000 | 0.151530 |
| train | cond_state_shuffle | 0.413750 | 2.885000 | 0.413750 | -0.026781 |
| train | cond_state_zero | 0.396250 | 2.993750 | 0.396250 | -0.018186 |
| val | original | 0.900000 | 1.260000 | 0.900000 | 0.086014 |
| val | action_zero | 0.900000 | 1.260000 | 0.900000 | 0.085797 |
| val | cond_state_shuffle | 0.360000 | 2.930000 | 0.360000 | -0.023762 |
| val | cond_state_zero | 0.390000 | 3.070000 | 0.390000 | -0.021471 |
| test | original | 0.890000 | 1.240000 | 0.890000 | 0.086292 |
| test | action_zero | 0.890000 | 1.240000 | 0.890000 | 0.086145 |
| test | cond_state_shuffle | 0.350000 | 3.020000 | 0.350000 | -0.039834 |
| test | cond_state_zero | 0.330000 | 3.170000 | 0.330000 | -0.023276 |

## Interpretation

This ranks predicted latent displacements against candidate latent displacements, removing the absolute-state shortcut from future-space ranking.

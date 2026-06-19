# Same-action delta-candidate MLP: state_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/same_action_hard/same_action_hard_v0_groups.npz`
- mode: `state_only`
- groups count: `1000`
- candidates: `5`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.910000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 1.000000 | 0.157439 |
| train | action_zero | 1.000000 | 1.000000 | 1.000000 | 0.157439 |
| train | cond_state_shuffle | 0.418750 | 2.877500 | 0.418750 | -0.028421 |
| train | cond_state_zero | 0.391250 | 2.993750 | 0.391250 | -0.018291 |
| val | original | 0.910000 | 1.280000 | 0.910000 | 0.088040 |
| val | action_zero | 0.910000 | 1.280000 | 0.910000 | 0.088040 |
| val | cond_state_shuffle | 0.430000 | 2.870000 | 0.430000 | -0.026208 |
| val | cond_state_zero | 0.390000 | 3.030000 | 0.390000 | -0.021281 |
| test | original | 0.930000 | 1.170000 | 0.930000 | 0.093772 |
| test | action_zero | 0.930000 | 1.170000 | 0.930000 | 0.093772 |
| test | cond_state_shuffle | 0.370000 | 3.110000 | 0.370000 | -0.044118 |
| test | cond_state_zero | 0.330000 | 3.150000 | 0.330000 | -0.023331 |

## Interpretation

This ranks predicted latent displacements against candidate latent displacements, removing the absolute-state shortcut from future-space ranking.

# Same-action delta-candidate MLP: action_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/same_action_hard/same_action_hard_v0_groups.npz`
- mode: `action_only`
- groups count: `1000`
- candidates: `5`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.240000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.275000 | 2.948750 | 0.275000 | -0.017213 |
| train | action_zero | 0.277500 | 2.963750 | 0.277500 | -0.017599 |
| train | cond_state_shuffle | 0.275000 | 2.948750 | 0.275000 | -0.017213 |
| train | cond_state_zero | 0.275000 | 2.948750 | 0.275000 | -0.017213 |
| val | original | 0.240000 | 3.130000 | 0.240000 | -0.020912 |
| val | action_zero | 0.220000 | 3.200000 | 0.220000 | -0.020734 |
| val | cond_state_shuffle | 0.240000 | 3.130000 | 0.240000 | -0.020912 |
| val | cond_state_zero | 0.240000 | 3.130000 | 0.240000 | -0.020912 |
| test | original | 0.270000 | 2.850000 | 0.270000 | -0.020461 |
| test | action_zero | 0.260000 | 2.930000 | 0.260000 | -0.020712 |
| test | cond_state_shuffle | 0.270000 | 2.850000 | 0.270000 | -0.020461 |
| test | cond_state_zero | 0.270000 | 2.850000 | 0.270000 | -0.020461 |

## Interpretation

This ranks predicted latent displacements against candidate latent displacements, removing the absolute-state shortcut from future-space ranking.

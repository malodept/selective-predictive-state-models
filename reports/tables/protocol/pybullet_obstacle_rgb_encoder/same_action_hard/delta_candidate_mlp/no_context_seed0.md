# Same-action delta-candidate MLP: no_context

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/same_action_hard/same_action_hard_v0_groups.npz`
- mode: `no_context`
- groups count: `1000`
- candidates: `5`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.250000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.357500 | 2.750000 | 0.357500 | -0.011288 |
| train | action_zero | 0.357500 | 2.750000 | 0.357500 | -0.011288 |
| train | cond_state_shuffle | 0.357500 | 2.750000 | 0.357500 | -0.011288 |
| train | cond_state_zero | 0.357500 | 2.750000 | 0.357500 | -0.011288 |
| val | original | 0.250000 | 3.170000 | 0.250000 | -0.019014 |
| val | action_zero | 0.250000 | 3.170000 | 0.250000 | -0.019014 |
| val | cond_state_shuffle | 0.250000 | 3.170000 | 0.250000 | -0.019014 |
| val | cond_state_zero | 0.250000 | 3.170000 | 0.250000 | -0.019014 |
| test | original | 0.260000 | 3.080000 | 0.260000 | -0.018858 |
| test | action_zero | 0.260000 | 3.080000 | 0.260000 | -0.018858 |
| test | cond_state_shuffle | 0.260000 | 3.080000 | 0.260000 | -0.018858 |
| test | cond_state_zero | 0.260000 | 3.080000 | 0.260000 | -0.018858 |

## Interpretation

This ranks predicted latent displacements against candidate latent displacements, removing the absolute-state shortcut from future-space ranking.

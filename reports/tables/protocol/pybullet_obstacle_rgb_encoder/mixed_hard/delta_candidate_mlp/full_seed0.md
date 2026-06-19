# Same-action delta-candidate MLP: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz`
- mode: `full`
- groups count: `1000`
- candidates: `5`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.380000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 1.000000 | 0.103191 |
| train | action_zero | 0.827500 | 1.196250 | 0.827500 | 0.052281 |
| train | cond_state_shuffle | 0.298750 | 2.733750 | 0.298750 | -0.053311 |
| train | cond_state_zero | 0.167500 | 2.948750 | 0.167500 | -0.031641 |
| val | original | 0.380000 | 2.160000 | 0.380000 | -0.034110 |
| val | action_zero | 0.140000 | 2.700000 | 0.140000 | -0.074312 |
| val | cond_state_shuffle | 0.250000 | 2.960000 | 0.250000 | -0.065226 |
| val | cond_state_zero | 0.060000 | 3.270000 | 0.060000 | -0.039343 |
| test | original | 0.390000 | 2.040000 | 0.390000 | -0.018460 |
| test | action_zero | 0.240000 | 2.480000 | 0.240000 | -0.057175 |
| test | cond_state_shuffle | 0.260000 | 2.910000 | 0.260000 | -0.067833 |
| test | cond_state_zero | 0.150000 | 3.170000 | 0.150000 | -0.033025 |

## Interpretation

This ranks predicted latent displacements against candidate latent displacements, removing the absolute-state shortcut from future-space ranking.

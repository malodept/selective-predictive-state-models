# Same-action delta-candidate MLP: no_context

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz`
- mode: `no_context`
- groups count: `1000`
- candidates: `5`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.230000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.396250 | 2.343750 | 0.396250 | -0.006765 |
| train | action_zero | 0.396250 | 2.343750 | 0.396250 | -0.006765 |
| train | cond_state_shuffle | 0.396250 | 2.343750 | 0.396250 | -0.006765 |
| train | cond_state_zero | 0.396250 | 2.343750 | 0.396250 | -0.006765 |
| val | original | 0.230000 | 3.160000 | 0.230000 | -0.027939 |
| val | action_zero | 0.230000 | 3.160000 | 0.230000 | -0.027939 |
| val | cond_state_shuffle | 0.230000 | 3.160000 | 0.230000 | -0.027939 |
| val | cond_state_zero | 0.230000 | 3.160000 | 0.230000 | -0.027939 |
| test | original | 0.270000 | 2.960000 | 0.270000 | -0.024259 |
| test | action_zero | 0.270000 | 2.960000 | 0.270000 | -0.024259 |
| test | cond_state_shuffle | 0.270000 | 2.960000 | 0.270000 | -0.024259 |
| test | cond_state_zero | 0.270000 | 2.960000 | 0.270000 | -0.024259 |

## Interpretation

This ranks predicted latent displacements against candidate latent displacements, removing the absolute-state shortcut from future-space ranking.

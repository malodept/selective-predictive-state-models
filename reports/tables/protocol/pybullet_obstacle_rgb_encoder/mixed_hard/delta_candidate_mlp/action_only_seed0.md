# Same-action delta-candidate MLP: action_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz`
- mode: `action_only`
- groups count: `1000`
- candidates: `5`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.340000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.576250 | 1.685000 | 0.576250 | 0.004703 |
| train | action_zero | 0.133750 | 3.177500 | 0.133750 | -0.035290 |
| train | cond_state_shuffle | 0.576250 | 1.685000 | 0.576250 | 0.004703 |
| train | cond_state_zero | 0.576250 | 1.685000 | 0.576250 | 0.004703 |
| val | original | 0.340000 | 2.240000 | 0.340000 | -0.022225 |
| val | action_zero | 0.050000 | 3.460000 | 0.050000 | -0.045135 |
| val | cond_state_shuffle | 0.340000 | 2.240000 | 0.340000 | -0.022225 |
| val | cond_state_zero | 0.340000 | 2.240000 | 0.340000 | -0.022225 |
| test | original | 0.310000 | 2.110000 | 0.310000 | -0.014952 |
| test | action_zero | 0.140000 | 3.400000 | 0.140000 | -0.038769 |
| test | cond_state_shuffle | 0.310000 | 2.110000 | 0.310000 | -0.014952 |
| test | cond_state_zero | 0.310000 | 2.110000 | 0.310000 | -0.014952 |

## Interpretation

This ranks predicted latent displacements against candidate latent displacements, removing the absolute-state shortcut from future-space ranking.

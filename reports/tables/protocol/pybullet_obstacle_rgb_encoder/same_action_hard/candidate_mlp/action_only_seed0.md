# Same-action hard-negative candidate MLP: action_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/same_action_hard/same_action_hard_v0_groups.npz`
- mode: `action_only`
- groups count: `1000`
- candidates: `5`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `0.990000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 1.000000 | 0.112849 |
| train | action_zero | 0.998750 | 1.001250 | 0.998750 | 0.112193 |
| train | cond_state_shuffle | 1.000000 | 1.000000 | 1.000000 | 0.112849 |
| train | cond_state_zero | 1.000000 | 1.000000 | 1.000000 | 0.112849 |
| val | original | 0.990000 | 1.010000 | 0.990000 | 0.107108 |
| val | action_zero | 0.990000 | 1.020000 | 0.990000 | 0.106713 |
| val | cond_state_shuffle | 0.990000 | 1.010000 | 0.990000 | 0.107108 |
| val | cond_state_zero | 0.990000 | 1.010000 | 0.990000 | 0.107108 |
| test | original | 0.970000 | 1.030000 | 0.970000 | 0.111635 |
| test | action_zero | 0.970000 | 1.030000 | 0.970000 | 0.111902 |
| test | cond_state_shuffle | 0.970000 | 1.030000 | 0.970000 | 0.111635 |
| test | cond_state_zero | 0.970000 | 1.030000 | 0.970000 | 0.111635 |

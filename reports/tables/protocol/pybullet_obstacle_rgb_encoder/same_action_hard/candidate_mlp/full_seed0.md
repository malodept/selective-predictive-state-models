# Same-action hard-negative candidate MLP: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/same_action_hard/same_action_hard_v0_groups.npz`
- mode: `full`
- groups count: `1000`
- candidates: `5`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best val top-1: `1.000000`

| split | variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 1.000000 | 0.152428 |
| train | action_zero | 1.000000 | 1.000000 | 1.000000 | 0.152401 |
| train | cond_state_shuffle | 0.970000 | 1.045000 | 0.970000 | 0.110395 |
| train | cond_state_zero | 0.993750 | 1.008750 | 0.993750 | 0.111198 |
| val | original | 1.000000 | 1.000000 | 1.000000 | 0.132619 |
| val | action_zero | 1.000000 | 1.000000 | 1.000000 | 0.132725 |
| val | cond_state_shuffle | 0.980000 | 1.030000 | 0.980000 | 0.104333 |
| val | cond_state_zero | 0.980000 | 1.030000 | 0.980000 | 0.104571 |
| test | original | 0.970000 | 1.030000 | 0.970000 | 0.143401 |
| test | action_zero | 0.970000 | 1.030000 | 0.970000 | 0.143394 |
| test | cond_state_shuffle | 0.970000 | 1.030000 | 0.970000 | 0.108754 |
| test | cond_state_zero | 0.970000 | 1.040000 | 0.970000 | 0.110576 |

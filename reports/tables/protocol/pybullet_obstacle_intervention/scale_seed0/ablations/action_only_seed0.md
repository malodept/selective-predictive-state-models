# Exact-intervention ablation MLP: action_only

- data: `outputs/counterfactual/pybullet_obstacles_scale/pybullet_obstacle_intervention_v1_2k_seed0.npz`
- mode: `action_only`
- groups: `2000`
- candidates: `5`
- train/val/test groups: `1600/200/200`
- chance top-1: `0.200000`
- best validation future MSE: `0.00458287`

## Candidate-matching evaluation

| split | intervention | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.211125 | 2.807375 | 0.211125 | -0.003976 |
| train | zero | 0.200000 | 3.000000 | 0.200000 | -0.005128 |
| train | within_group_shuffle | 0.197625 | 2.997500 | 0.197625 | -0.004990 |
| train | within_group_reverse | 0.198250 | 2.944625 | 0.198250 | -0.004814 |
| val | original | 0.214000 | 2.822000 | 0.214000 | -0.003882 |
| val | zero | 0.200000 | 3.000000 | 0.200000 | -0.005038 |
| val | within_group_shuffle | 0.202000 | 3.007000 | 0.202000 | -0.004916 |
| val | within_group_reverse | 0.190000 | 2.959000 | 0.190000 | -0.004715 |
| test | original | 0.210000 | 2.812000 | 0.210000 | -0.003900 |
| test | zero | 0.200000 | 3.000000 | 0.200000 | -0.004988 |
| test | within_group_shuffle | 0.197000 | 3.004000 | 0.197000 | -0.004899 |
| test | within_group_reverse | 0.199000 | 2.934000 | 0.199000 | -0.004693 |

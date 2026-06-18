# Exact-intervention ablation MLP: full

- data: `outputs/counterfactual/pybullet_obstacles_scale/pybullet_obstacle_intervention_v1_2k_seed0.npz`
- mode: `full`
- groups: `2000`
- candidates: `5`
- train/val/test groups: `1600/200/200`
- chance top-1: `0.200000`
- best validation future MSE: `0.00036601`

## Candidate-matching evaluation

| split | intervention | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.973875 | 1.040500 | 0.973875 | 0.004185 |
| train | zero | 0.200000 | 3.000000 | 0.200000 | -0.005133 |
| train | within_group_shuffle | 0.207000 | 2.971750 | 0.207000 | -0.005741 |
| train | within_group_reverse | 0.204625 | 2.875250 | 0.204625 | -0.004925 |
| val | original | 0.977000 | 1.034000 | 0.977000 | 0.004107 |
| val | zero | 0.200000 | 3.000000 | 0.200000 | -0.005057 |
| val | within_group_shuffle | 0.187000 | 3.007000 | 0.187000 | -0.005828 |
| val | within_group_reverse | 0.202000 | 2.889000 | 0.202000 | -0.004853 |
| test | original | 0.960000 | 1.064000 | 0.960000 | 0.003974 |
| test | zero | 0.200000 | 3.000000 | 0.200000 | -0.004999 |
| test | within_group_shuffle | 0.189000 | 3.039000 | 0.189000 | -0.005896 |
| test | within_group_reverse | 0.206000 | 2.897000 | 0.206000 | -0.004900 |

# Exact-intervention ablation MLP: full

- data: `outputs/counterfactual/pybullet_obstacles/pybullet_obstacle_intervention_v1_pilot.npz`
- mode: `full`
- groups: `500`
- candidates: `5`
- train/val/test groups: `400/50/50`
- chance top-1: `0.200000`
- best validation future MSE: `0.00123776`

## Candidate-matching evaluation

| split | intervention | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.880000 | 1.207000 | 0.880000 | 0.002369 |
| train | zero | 0.200000 | 3.000000 | 0.200000 | -0.005114 |
| train | within_group_shuffle | 0.196000 | 2.984000 | 0.196000 | -0.004733 |
| train | within_group_reverse | 0.229500 | 2.839500 | 0.229500 | -0.004133 |
| val | original | 0.856000 | 1.280000 | 0.856000 | 0.002364 |
| val | zero | 0.200000 | 3.000000 | 0.200000 | -0.004907 |
| val | within_group_shuffle | 0.156000 | 3.120000 | 0.156000 | -0.005084 |
| val | within_group_reverse | 0.240000 | 2.800000 | 0.240000 | -0.004297 |
| test | original | 0.852000 | 1.272000 | 0.852000 | 0.002154 |
| test | zero | 0.200000 | 3.000000 | 0.200000 | -0.005250 |
| test | within_group_shuffle | 0.172000 | 3.108000 | 0.172000 | -0.005149 |
| test | within_group_reverse | 0.256000 | 2.752000 | 0.256000 | -0.003989 |

# Exact-intervention ablation MLP: no_context

- data: `outputs/counterfactual/pybullet_obstacles_scale/pybullet_obstacle_intervention_v1_2k_seed0.npz`
- mode: `no_context`
- groups: `2000`
- candidates: `5`
- train/val/test groups: `1600/200/200`
- chance top-1: `0.200000`
- best validation future MSE: `0.00509070`

## Candidate-matching evaluation

| split | intervention | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.200000 | 3.000000 | 0.200000 | -0.004991 |
| train | zero | 0.200000 | 3.000000 | 0.200000 | -0.004991 |
| train | within_group_shuffle | 0.200000 | 3.000000 | 0.200000 | -0.004991 |
| train | within_group_reverse | 0.200000 | 3.000000 | 0.200000 | -0.004991 |
| val | original | 0.200000 | 3.000000 | 0.200000 | -0.004907 |
| val | zero | 0.200000 | 3.000000 | 0.200000 | -0.004907 |
| val | within_group_shuffle | 0.200000 | 3.000000 | 0.200000 | -0.004907 |
| val | within_group_reverse | 0.200000 | 3.000000 | 0.200000 | -0.004907 |
| test | original | 0.200000 | 3.000000 | 0.200000 | -0.004844 |
| test | zero | 0.200000 | 3.000000 | 0.200000 | -0.004844 |
| test | within_group_shuffle | 0.200000 | 3.000000 | 0.200000 | -0.004844 |
| test | within_group_reverse | 0.200000 | 3.000000 | 0.200000 | -0.004844 |

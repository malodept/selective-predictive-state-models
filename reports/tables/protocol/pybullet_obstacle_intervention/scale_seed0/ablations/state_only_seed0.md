# Exact-intervention ablation MLP: state_only

- data: `outputs/counterfactual/pybullet_obstacles_scale/pybullet_obstacle_intervention_v1_2k_seed0.npz`
- mode: `state_only`
- groups: `2000`
- candidates: `5`
- train/val/test groups: `1600/200/200`
- chance top-1: `0.200000`
- best validation future MSE: `0.00373666`

## Candidate-matching evaluation

| split | intervention | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.200000 | 3.000000 | 0.200000 | -0.002501 |
| train | zero | 0.200000 | 3.000000 | 0.200000 | -0.002501 |
| train | within_group_shuffle | 0.200000 | 3.000000 | 0.200000 | -0.002501 |
| train | within_group_reverse | 0.200000 | 3.000000 | 0.200000 | -0.002501 |
| val | original | 0.200000 | 3.000000 | 0.200000 | -0.002479 |
| val | zero | 0.200000 | 3.000000 | 0.200000 | -0.002479 |
| val | within_group_shuffle | 0.200000 | 3.000000 | 0.200000 | -0.002479 |
| val | within_group_reverse | 0.200000 | 3.000000 | 0.200000 | -0.002479 |
| test | original | 0.200000 | 3.000000 | 0.200000 | -0.002389 |
| test | zero | 0.200000 | 3.000000 | 0.200000 | -0.002389 |
| test | within_group_shuffle | 0.200000 | 3.000000 | 0.200000 | -0.002389 |
| test | within_group_reverse | 0.200000 | 3.000000 | 0.200000 | -0.002389 |

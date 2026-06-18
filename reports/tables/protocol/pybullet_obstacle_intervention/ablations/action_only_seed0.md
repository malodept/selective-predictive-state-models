# Exact-intervention ablation MLP: action_only

- data: `outputs/counterfactual/pybullet_obstacles/pybullet_obstacle_intervention_v1_pilot.npz`
- mode: `action_only`
- groups: `500`
- candidates: `5`
- train/val/test groups: `400/50/50`
- chance top-1: `0.200000`
- best validation future MSE: `0.00468948`

## Candidate-matching evaluation

| split | intervention | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.213000 | 2.800500 | 0.213000 | -0.003943 |
| train | zero | 0.200000 | 3.000000 | 0.200000 | -0.005039 |
| train | within_group_shuffle | 0.199000 | 3.008500 | 0.199000 | -0.004946 |
| train | within_group_reverse | 0.194000 | 2.971500 | 0.194000 | -0.004739 |
| val | original | 0.204000 | 2.840000 | 0.204000 | -0.003930 |
| val | zero | 0.200000 | 3.000000 | 0.200000 | -0.004934 |
| val | within_group_shuffle | 0.196000 | 2.988000 | 0.196000 | -0.004773 |
| val | within_group_reverse | 0.200000 | 2.928000 | 0.200000 | -0.004605 |
| test | original | 0.208000 | 2.836000 | 0.208000 | -0.004167 |
| test | zero | 0.200000 | 3.000000 | 0.200000 | -0.005177 |
| test | within_group_shuffle | 0.204000 | 3.000000 | 0.204000 | -0.005234 |
| test | within_group_reverse | 0.200000 | 2.944000 | 0.200000 | -0.004847 |

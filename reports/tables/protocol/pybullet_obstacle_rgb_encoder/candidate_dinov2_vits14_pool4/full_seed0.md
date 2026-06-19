# Exact-intervention candidate MLP: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- mode: `full`
- groups: `500`
- candidates: `5`
- train/val/test groups: `400/50/50`
- chance top-1: `0.200000`
- best val top-1: `1.000000`
- temperature: `0.02`
- lambda MSE: `0.01`

## Candidate-matching evaluation

| split | intervention | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 1.000000 | 0.094815 |
| train | zero | 0.200000 | 3.000000 | 0.200000 | -0.060657 |
| train | within_group_shuffle | 0.190000 | 3.009000 | 0.190000 | -0.111683 |
| train | within_group_reverse | 0.200000 | 2.656500 | 0.200000 | -0.084043 |
| val | original | 1.000000 | 1.000000 | 1.000000 | 0.090087 |
| val | zero | 0.200000 | 3.000000 | 0.200000 | -0.061371 |
| val | within_group_shuffle | 0.156000 | 3.000000 | 0.156000 | -0.115599 |
| val | within_group_reverse | 0.200000 | 2.632000 | 0.200000 | -0.080274 |
| test | original | 0.988000 | 1.012000 | 0.988000 | 0.091668 |
| test | zero | 0.200000 | 3.000000 | 0.200000 | -0.061966 |
| test | within_group_shuffle | 0.160000 | 3.100000 | 0.160000 | -0.121580 |
| test | within_group_reverse | 0.196000 | 2.532000 | 0.196000 | -0.078007 |

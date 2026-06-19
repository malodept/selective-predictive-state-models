# Exact-intervention candidate MLP: action_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- mode: `action_only`
- groups: `500`
- candidates: `5`
- train/val/test groups: `400/50/50`
- chance top-1: `0.200000`
- best val top-1: `0.996000`
- temperature: `0.02`
- lambda MSE: `0.01`

## Candidate-matching evaluation

| split | intervention | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 1.000000 | 0.131432 |
| train | zero | 0.200000 | 3.000000 | 0.200000 | -0.103145 |
| train | within_group_shuffle | 0.190000 | 3.018500 | 0.190000 | -0.133025 |
| train | within_group_reverse | 0.200000 | 2.951500 | 0.200000 | -0.120278 |
| val | original | 0.996000 | 1.004000 | 0.996000 | 0.129799 |
| val | zero | 0.200000 | 3.000000 | 0.200000 | -0.103160 |
| val | within_group_shuffle | 0.156000 | 3.068000 | 0.156000 | -0.141152 |
| val | within_group_reverse | 0.204000 | 2.976000 | 0.204000 | -0.115678 |
| test | original | 0.996000 | 1.004000 | 0.996000 | 0.126911 |
| test | zero | 0.200000 | 3.000000 | 0.200000 | -0.103655 |
| test | within_group_shuffle | 0.156000 | 3.188000 | 0.156000 | -0.148139 |
| test | within_group_reverse | 0.196000 | 2.804000 | 0.196000 | -0.110665 |

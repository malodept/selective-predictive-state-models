# Exact-intervention candidate MLP: state_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- mode: `state_only`
- groups: `500`
- candidates: `5`
- train/val/test groups: `400/50/50`
- chance top-1: `0.200000`
- best val top-1: `0.200000`
- temperature: `0.02`
- lambda MSE: `0.01`

## Candidate-matching evaluation

| split | intervention | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.200000 | 3.000000 | 0.200000 | -0.033045 |
| train | zero | 0.200000 | 3.000000 | 0.200000 | -0.033045 |
| train | within_group_shuffle | 0.200000 | 3.000000 | 0.200000 | -0.033045 |
| train | within_group_reverse | 0.200000 | 3.000000 | 0.200000 | -0.033045 |
| val | original | 0.200000 | 3.000000 | 0.200000 | -0.033345 |
| val | zero | 0.200000 | 3.000000 | 0.200000 | -0.033345 |
| val | within_group_shuffle | 0.200000 | 3.000000 | 0.200000 | -0.033345 |
| val | within_group_reverse | 0.200000 | 3.000000 | 0.200000 | -0.033345 |
| test | original | 0.200000 | 3.000000 | 0.200000 | -0.034569 |
| test | zero | 0.200000 | 3.000000 | 0.200000 | -0.034569 |
| test | within_group_shuffle | 0.200000 | 3.000000 | 0.200000 | -0.034569 |
| test | within_group_reverse | 0.200000 | 3.000000 | 0.200000 | -0.034569 |

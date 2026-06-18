# Exact-intervention ablation MLP: full

- data: `outputs/counterfactual/pybullet_exact/pybullet_exact_intervention_v0_pilot.npz`
- mode: `full`
- groups: `500`
- candidates: `5`
- train/val/test groups: `400/50/50`
- chance top-1: `0.200000`
- best validation future MSE: `0.00079370`

## Candidate-matching evaluation

| split | intervention | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.999500 | 1.000500 | 0.999500 | 0.006606 |
| train | zero | 0.200000 | 3.000000 | 0.200000 | -0.007530 |
| train | within_group_shuffle | 0.199000 | 2.969000 | 0.199000 | -0.009079 |
| train | within_group_reverse | 0.200000 | 2.844500 | 0.200000 | -0.007242 |
| val | original | 1.000000 | 1.000000 | 1.000000 | 0.007037 |
| val | zero | 0.200000 | 3.000000 | 0.200000 | -0.007769 |
| val | within_group_shuffle | 0.156000 | 3.072000 | 0.156000 | -0.010043 |
| val | within_group_reverse | 0.200000 | 2.824000 | 0.200000 | -0.007277 |
| test | original | 0.992000 | 1.012000 | 0.992000 | 0.006307 |
| test | zero | 0.200000 | 3.000000 | 0.200000 | -0.007393 |
| test | within_group_shuffle | 0.160000 | 3.044000 | 0.160000 | -0.009214 |
| test | within_group_reverse | 0.200000 | 2.860000 | 0.200000 | -0.007423 |

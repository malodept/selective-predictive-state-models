# Exact-intervention ablation MLP: action_only

- data: `outputs/counterfactual/pybullet_exact/pybullet_exact_intervention_v0_pilot.npz`
- mode: `action_only`
- groups: `500`
- candidates: `5`
- train/val/test groups: `400/50/50`
- chance top-1: `0.200000`
- best validation future MSE: `0.00749112`

## Candidate-matching evaluation

| split | intervention | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.200000 | 2.860000 | 0.200000 | -0.006021 |
| train | zero | 0.200000 | 3.000000 | 0.200000 | -0.007580 |
| train | within_group_shuffle | 0.200000 | 2.986500 | 0.200000 | -0.007450 |
| train | within_group_reverse | 0.200000 | 2.947000 | 0.200000 | -0.007101 |
| val | original | 0.200000 | 2.860000 | 0.200000 | -0.006085 |
| val | zero | 0.200000 | 3.000000 | 0.200000 | -0.007976 |
| val | within_group_shuffle | 0.200000 | 3.020000 | 0.200000 | -0.007912 |
| val | within_group_reverse | 0.200000 | 2.984000 | 0.200000 | -0.007342 |
| test | original | 0.200000 | 2.852000 | 0.200000 | -0.005954 |
| test | zero | 0.200000 | 3.000000 | 0.200000 | -0.007469 |
| test | within_group_shuffle | 0.200000 | 3.012000 | 0.200000 | -0.007355 |
| test | within_group_reverse | 0.200000 | 2.932000 | 0.200000 | -0.006932 |

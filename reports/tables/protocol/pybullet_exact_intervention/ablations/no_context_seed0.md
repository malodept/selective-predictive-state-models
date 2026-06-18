# Exact-intervention ablation MLP: no_context

- data: `outputs/counterfactual/pybullet_exact/pybullet_exact_intervention_v0_pilot.npz`
- mode: `no_context`
- groups: `500`
- candidates: `5`
- train/val/test groups: `400/50/50`
- chance top-1: `0.200000`
- best validation future MSE: `0.00850333`

## Candidate-matching evaluation

| split | intervention | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 0.200000 | 3.000000 | 0.200000 | -0.007440 |
| train | zero | 0.200000 | 3.000000 | 0.200000 | -0.007440 |
| train | within_group_shuffle | 0.200000 | 3.000000 | 0.200000 | -0.007440 |
| train | within_group_reverse | 0.200000 | 3.000000 | 0.200000 | -0.007440 |
| val | original | 0.200000 | 3.000000 | 0.200000 | -0.007842 |
| val | zero | 0.200000 | 3.000000 | 0.200000 | -0.007842 |
| val | within_group_shuffle | 0.200000 | 3.000000 | 0.200000 | -0.007842 |
| val | within_group_reverse | 0.200000 | 3.000000 | 0.200000 | -0.007842 |
| test | original | 0.200000 | 3.000000 | 0.200000 | -0.007316 |
| test | zero | 0.200000 | 3.000000 | 0.200000 | -0.007316 |
| test | within_group_shuffle | 0.200000 | 3.000000 | 0.200000 | -0.007316 |
| test | within_group_reverse | 0.200000 | 3.000000 | 0.200000 | -0.007316 |

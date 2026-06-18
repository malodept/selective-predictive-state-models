# Toy exact-intervention action MLP

- data: `outputs/counterfactual/toy_exact/toy_exact_intervention_v0.npz`
- groups: `1000`
- candidates: `5`
- train/val/test groups: `800/100/100`
- chance top-1: `0.200000`
- best validation future MSE: `0.00081975`

## Candidate-matching evaluation

| split | mode | top-1 | mean rank | positive margin frac | mean margin |
| --- | --- | ---: | ---: | ---: | ---: |
| train | original | 1.000000 | 1.000000 | 1.000000 | 0.008573 |
| train | zero | 0.200000 | 3.000000 | 0.200000 | -0.007061 |
| train | within_group_shuffle | 0.204500 | 2.979000 | 0.204500 | -0.009444 |
| train | within_group_reverse | 0.200000 | 2.803250 | 0.200000 | -0.008300 |
| val | original | 1.000000 | 1.000000 | 1.000000 | 0.008037 |
| val | zero | 0.200000 | 3.000000 | 0.200000 | -0.007077 |
| val | within_group_shuffle | 0.148000 | 3.068000 | 0.148000 | -0.009625 |
| val | within_group_reverse | 0.200000 | 2.806000 | 0.200000 | -0.007974 |
| test | original | 1.000000 | 1.000000 | 1.000000 | 0.008026 |
| test | zero | 0.200000 | 3.000000 | 0.200000 | -0.007024 |
| test | within_group_shuffle | 0.148000 | 3.054000 | 0.148000 | -0.010127 |
| test | within_group_reverse | 0.200000 | 2.798000 | 0.200000 | -0.008160 |

## Interpretation rule

A successful exact-intervention learner should have test `original` far above chance and clearly above zero/shuffle interventions.
This validates that action-grounded candidate matching is learnable when the data truly contains exact branches from the same state.

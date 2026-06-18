# Toy exact-intervention baseline evaluation: toy_exact_intervention_v0

- data: `outputs/counterfactual/toy_exact/toy_exact_intervention_v0.npz`
- groups: `1000`
- candidates: `5`
- chance top-1: `0.200000`

| model | alpha | top-1 | mean rank | positive margin frac | mean margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.007147 |
| delta_oracle_original | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.007147 |
| delta_oracle_original | 0.25 | 0.200000 | 2.039200 | 0.200000 | -0.002851 |
| delta_oracle_original | 0.50 | 0.535200 | 1.464800 | 0.535200 | 0.001444 |
| delta_oracle_original | 0.75 | 1.000000 | 1.000000 | 1.000000 | 0.005660 |
| delta_oracle_original | 1.00 | 1.000000 | 1.000000 | 1.000000 | 0.009827 |
| delta_oracle_deranged | 1.00 | 0.000000 | 3.512400 | 0.000000 | -0.015335 |
| delta_oracle_random_shuffle | 1.00 | 0.206000 | 3.004800 | 0.206000 | -0.010184 |

## Expected outcome

The identity baseline should be near chance. The original delta oracle should reach top-1 = 1.0 at alpha = 1.
The deranged oracle should fail because the action-future correspondence is deliberately wrong.
This verifies that the candidate-matching metric can detect true exact-intervention grounding when the protocol is identifiable.

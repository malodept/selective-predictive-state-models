# Exact-intervention baseline evaluation: pybullet_exact_intervention_v0_pilot

- data: `outputs/counterfactual/pybullet_exact/pybullet_exact_intervention_v0_pilot.npz`
- groups: `500`
- candidates: `5`
- chance top-1: `0.200000`

| model | alpha | top-1 | mean rank | positive margin frac | mean margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.007607 |
| delta_oracle_original | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.007607 |
| delta_oracle_original | 0.25 | 0.200000 | 2.615600 | 0.200000 | -0.003567 |
| delta_oracle_original | 0.50 | 0.357200 | 1.819200 | 0.357200 | 0.000343 |
| delta_oracle_original | 0.75 | 1.000000 | 1.000000 | 1.000000 | 0.004102 |
| delta_oracle_original | 1.00 | 1.000000 | 1.000000 | 1.000000 | 0.007779 |
| delta_oracle_deranged | 1.00 | 0.000000 | 3.495600 | 0.000000 | -0.014497 |
| delta_oracle_random_shuffle | 1.00 | 0.216000 | 2.942800 | 0.216000 | -0.009579 |

## Expected outcome

A valid exact-intervention protocol should make the original delta oracle clearly outperform identity and random/shuffled baselines.
If the oracle is weak, the representation or rendering does not make the intervention branches identifiable enough.

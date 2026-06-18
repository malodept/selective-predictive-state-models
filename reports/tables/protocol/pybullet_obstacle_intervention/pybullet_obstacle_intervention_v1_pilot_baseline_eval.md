# Exact-intervention baseline evaluation: pybullet_obstacle_intervention_v1_pilot

- data: `outputs/counterfactual/pybullet_obstacles/pybullet_obstacle_intervention_v1_pilot.npz`
- groups: `500`
- candidates: `5`
- chance top-1: `0.200000`

| model | alpha | top-1 | mean rank | positive margin frac | mean margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.005055 |
| delta_oracle_original | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.005055 |
| delta_oracle_original | 0.25 | 0.200000 | 2.620000 | 0.200000 | -0.002461 |
| delta_oracle_original | 0.50 | 0.346000 | 1.884800 | 0.346000 | 0.000068 |
| delta_oracle_original | 0.75 | 1.000000 | 1.000000 | 1.000000 | 0.002512 |
| delta_oracle_original | 1.00 | 1.000000 | 1.000000 | 1.000000 | 0.004878 |
| delta_oracle_deranged | 1.00 | 0.000000 | 3.475600 | 0.000000 | -0.009000 |
| delta_oracle_random_shuffle | 1.00 | 0.216000 | 2.981600 | 0.216000 | -0.006152 |

## Expected outcome

A valid exact-intervention protocol should make the original delta oracle clearly outperform identity and random/shuffled baselines.
If the oracle is weak, the representation or rendering does not make the intervention branches identifiable enough.

# Exact-intervention baseline evaluation: pybullet_obstacle_intervention_v1_2k_seed0

- data: `outputs/counterfactual/pybullet_obstacles_scale/pybullet_obstacle_intervention_v1_2k_seed0.npz`
- groups: `2000`
- candidates: `5`
- chance top-1: `0.200000`

| model | alpha | top-1 | mean rank | positive margin frac | mean margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.005097 |
| delta_oracle_original | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.005097 |
| delta_oracle_original | 0.25 | 0.200000 | 2.620300 | 0.200000 | -0.002482 |
| delta_oracle_original | 0.50 | 0.351900 | 1.886900 | 0.351900 | 0.000066 |
| delta_oracle_original | 0.75 | 1.000000 | 1.000000 | 1.000000 | 0.002526 |
| delta_oracle_original | 1.00 | 1.000000 | 1.000000 | 1.000000 | 0.004914 |
| delta_oracle_deranged | 1.00 | 0.000000 | 3.494200 | 0.000000 | -0.009056 |
| delta_oracle_random_shuffle | 1.00 | 0.197900 | 2.995100 | 0.197900 | -0.006281 |

## Expected outcome

A valid exact-intervention protocol should make the original delta oracle clearly outperform identity and random/shuffled baselines.
If the oracle is weak, the representation or rendering does not make the intervention branches identifiable enough.

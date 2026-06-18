# PyBullet obstacle v1 scale-up summary

Dataset: `pybullet_obstacle_intervention_v1_2k_seed0`.

| mode | best val MSE | test original top-1 | test zero top-1 | test shuffle top-1 | test reverse top-1 | original-zero gap | original-shuffle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.00036601 | 0.960000 | 0.200000 | 0.189000 | 0.206000 | 0.760000 | 0.771000 |
| action_only | 0.00458287 | 0.210000 | 0.200000 | 0.197000 | 0.199000 | 0.010000 | 0.013000 |
| state_only | 0.00373666 | 0.200000 | 0.200000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 |
| no_context | 0.00509070 | 0.200000 | 0.200000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 |

## Interpretation

The scale-up is successful if `full` remains far above chance while `action_only`, `state_only`, and `no_context` remain near chance.
This checks that the 500-group pilot was not a small-test artifact.

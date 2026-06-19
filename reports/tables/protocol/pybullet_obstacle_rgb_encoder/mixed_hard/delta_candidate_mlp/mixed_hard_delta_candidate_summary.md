# Mixed hard-negative delta-ranking summary

Dataset: `pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4`.

| mode | best val top-1 | test original | action zero | state shuffle | state zero | gain vs chance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.380000 | 0.390000 | 0.240000 | 0.260000 | 0.150000 | 0.190000 |
| action_only | 0.340000 | 0.310000 | 0.140000 | 0.310000 | 0.310000 | 0.110000 |
| state_only | 0.250000 | 0.250000 | 0.250000 | 0.250000 | 0.140000 | 0.050000 |
| no_context | 0.230000 | 0.270000 | 0.270000 | 0.270000 | 0.270000 | 0.070000 |

## Interpretation

This mixed protocol combines same-state/different-action negatives and same-action/different-state negatives.
A strong result requires the full state-action model to outperform both action-only and state-only.

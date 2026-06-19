# Same-action hard-negative delta-ranking summary

Dataset: `pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4`.

| mode | best val top-1 | test original | action zero | state shuffle | state zero | original - action_only chance gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.900000 | 0.890000 | 0.890000 | 0.350000 | 0.330000 | 0.690000 |
| action_only | 0.240000 | 0.270000 | 0.260000 | 0.270000 | 0.270000 | 0.070000 |
| state_only | 0.910000 | 0.930000 | 0.930000 | 0.370000 | 0.330000 | 0.730000 |
| no_context | 0.250000 | 0.260000 | 0.260000 | 0.260000 | 0.260000 | 0.060000 |

## Interpretation

This protocol ranks predicted latent displacements against same-action hard negatives.
It removes the absolute future shortcut and tests whether the model can use the current visual state to distinguish blocked from unblocked dynamics.

A successful result is: full clearly above chance, action-only near chance, and full degraded by state shuffling or state zeroing.

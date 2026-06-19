# Mixed hard-negative delta Transformer summary

Dataset: `pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4`.

| mode | best val top-1 | test original | action zero | state shuffle | state zero | gain vs chance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.980000 | 0.970000 | 0.200000 | 0.330000 | 0.320000 | 0.770000 |
| action_only | 0.310000 | 0.280000 | 0.280000 | 0.280000 | 0.280000 | 0.080000 |
| state_only | 0.310000 | 0.250000 | 0.250000 | 0.160000 | 0.110000 | 0.050000 |
| no_context | 0.220000 | 0.210000 | 0.210000 | 0.210000 | 0.210000 | 0.010000 |

## Interpretation

This experiment replaces the global MLP by a spatial Transformer over frozen DINOv2 patch tokens.
A strong result is obtained if the full model is far above action-only, state-only, and no-context, and if zeroing either action or state degrades performance.

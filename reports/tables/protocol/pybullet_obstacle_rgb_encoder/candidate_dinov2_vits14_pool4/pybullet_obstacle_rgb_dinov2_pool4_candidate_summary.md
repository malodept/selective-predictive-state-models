# PyBullet obstacle RGB DINOv2 pooled-4x4 candidate-objective summary

Dataset: `pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4`.

| mode | best val top-1 | test original top-1 | test zero top-1 | test shuffle top-1 | test reverse top-1 | original-zero gap | original-shuffle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 1.000000 | 0.988000 | 0.200000 | 0.160000 | 0.196000 | 0.788000 | 0.828000 |
| action_only | 0.996000 | 0.996000 | 0.200000 | 0.156000 | 0.196000 | 0.796000 | 0.840000 |
| state_only | 0.200000 | 0.200000 | 0.200000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 |
| no_context | 0.200000 | 0.200000 | 0.200000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 |

## Interpretation

This compares exact-intervention candidate matching on frozen DINOv2 patch-token features.
The experiment succeeds if the full state-action model is far above chance while action-only, state-only, and no-context remain near chance.

The key diagnostic is that MSE regression failed at chance on the same DINOv2 features, while the candidate-matching objective learned the action-grounded ranking.

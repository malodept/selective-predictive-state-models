# Mixed hard-negative delta Transformer multiseed summary

Dataset: `pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4`.

Values are mean ± sample standard deviation over seeds 0, 1, 2.

| mode | best val top-1 | test original | action zero | state shuffle | state zero | gain vs chance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.963333 ± 0.015275 | 0.953333 ± 0.015275 | 0.263333 ± 0.070946 | 0.410000 ± 0.069282 | 0.330000 ± 0.055678 | 0.753333 ± 0.015275 |
| action_only | 0.373333 ± 0.055076 | 0.293333 ± 0.051316 | 0.193333 ± 0.080829 | 0.293333 ± 0.051316 | 0.293333 ± 0.051316 | 0.093333 ± 0.051316 |
| state_only | 0.343333 ± 0.030551 | 0.250000 ± 0.040000 | 0.250000 ± 0.040000 | 0.176667 ± 0.028868 | 0.143333 ± 0.030551 | 0.050000 ± 0.040000 |
| no_context | 0.266667 ± 0.050332 | 0.190000 ± 0.026458 | 0.190000 ± 0.026458 | 0.190000 ± 0.026458 | 0.190000 ± 0.026458 | -0.010000 ± 0.026458 |

## Interpretation

The full spatial Transformer is robust across seeds. It remains far above chance and far above all ablations.

Zeroing the action or corrupting the state sharply reduces full-model performance, confirming that the model uses both intervention and visual state information.

The action-only, state-only, and no-context baselines remain close to chance relative to the full model.

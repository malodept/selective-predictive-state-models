# Mixed hard-negative delta Transformer pairwise multiseed summary

Values are mean ± sample standard deviation over seeds 0, 1, 2.

| mode | global top-1 | same-state / different-action | same-action / different-state |
| --- | ---: | ---: | ---: |
| full | 0.953333 ± 0.015275 | 0.988333 ± 0.005774 | 0.973333 ± 0.007638 |
| action_only | 0.293333 ± 0.051316 | 0.825000 ± 0.065574 | 0.463333 ± 0.044814 |
| state_only | 0.250000 ± 0.040000 | 0.515000 ± 0.052915 | 0.750000 ± 0.021794 |
| no_context | 0.190000 ± 0.026458 | 0.465000 ± 0.054083 | 0.461667 ± 0.051316 |

## Interpretation

The pairwise decomposition tests the two axes of the mixed protocol.

`same-state / different-action` requires action information.
`same-action / different-state` requires state-dependent visual dynamics.

The full model should be the only model clearly above chance on both axes.

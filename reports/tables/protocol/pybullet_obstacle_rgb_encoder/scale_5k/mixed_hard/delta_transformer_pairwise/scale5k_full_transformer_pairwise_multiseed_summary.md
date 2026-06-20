# Scale-5k full Transformer pairwise multiseed summary

Values are mean ± sample standard deviation over seeds 0, 1, 2.

| model | global top-1 | same-state / different-action | same-action / different-state |
| --- | ---: | ---: | ---: |
| full Transformer | 0.997667 ± 0.000577 | 0.999000 ± 0.000500 | 0.998667 ± 0.001155 |

## Interpretation

The full spatial Transformer remains near-perfect across both axes of the mixed hard-negative protocol.

`same-state / different-action` tests whether the model uses action information.
`same-action / different-state` tests whether the model uses state-dependent visual dynamics.

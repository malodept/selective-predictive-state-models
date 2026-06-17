# V-JEPA 2.1 action vs no-action multiseed summary

| model | seeds | identity error | raw error | global alpha | global error | gain vs identity | relative gain | cosine mean | positive cosine frac | pred Δ norm med | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V-JEPA 2.1 action | 3 | 0.031137 ± 0.000000 | 0.031496 ± 0.000588 | 0.573333 ± 0.030551 | 0.027374 ± 0.000056 | 0.003764 ± 0.000056 | 12.087491 ± 0.179279% | 0.307034 ± 0.003016 | 0.998817 ± 0.000362 | 12.687767 ± 0.525621 | 1.000000 ± 0.000000 |
| V-JEPA 2.1 no-action | 3 | 0.031137 ± 0.000000 | 0.029543 ± 0.000546 | 0.680000 ± 0.020000 | 0.027714 ± 0.000079 | 0.003423 ± 0.000079 | 10.994853 ± 0.254644% | 0.306508 ± 0.002301 | 0.998202 ± 0.000481 | 10.946400 ± 0.532067 | 1.000000 ± 0.000000 |

## Interpretation

- Mean action gain: `0.003764`.
- Mean no-action gain: `0.003423`.
- Mean action global error: `0.027374`.
- Mean no-action global error: `0.027714`.
- Action minus no-action gain: `0.000340`.
- No-action error minus action error: `0.000340`.

This table tests whether V-JEPA 2.1 makes action conditioning useful beyond latent video flow.

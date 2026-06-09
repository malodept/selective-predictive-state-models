# Error-supervised reliability multi-seed summary

| metric | mean | std | values |
| --- | ---: | ---: | --- |
| R@1 | 0.0688 | 0.0029 | 0.0655, 0.0684, 0.0726 |
| R@5 | 0.3186 | 0.0157 | 0.2965, 0.3311, 0.3282 |
| R@10 | 0.5474 | 0.0261 | 0.5105, 0.5643, 0.5675 |
| expected_learned_auroc | 0.7605 | 0.0031 | 0.7606, 0.7566, 0.7643 |
| expected_residual_auroc | 0.9832 | 0.0018 | 0.9827, 0.9813, 0.9857 |
| observed_residual_auroc | 0.9564 | 0.0022 | 0.9534, 0.9577, 0.9581 |
| observed_learned_auroc | 0.5120 | 0.0050 | 0.5163, 0.5049, 0.5147 |

## Best selector policy per seed

| seed | policy | error | compute | selected | utility |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | threshold=0.50 | 0.1972 | 1.4739 | 0.1580 | -0.2562 |
| 1 | threshold=0.65 | 0.1874 | 1.6015 | 0.2005 | -0.2515 |
| 2 | threshold=0.50 | 0.1838 | 1.6561 | 0.2187 | -0.2501 |

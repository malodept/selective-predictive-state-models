# Directional loss sweep summary

The directional residual loss is `MSE + lambda_cos * cosine_loss + 0.001 * lognorm_loss`.

## Multiseed comparison

| config | seeds | test global error | test improvement vs identity | test cosine mean | test cosine positive frac | test global alpha | test pred Δ norm med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| lambda_cos=0.01 | 3 | 1.589322 ± 0.000224 | 0.017654 ± 0.000224 | 0.088062 ± 0.000610 | 0.857781 ± 0.007760 | 0.165301 ± 0.008414 | 15.011401 ± 0.729456 |
| lambda_cos=0.05 | 3 | 1.589900 ± 0.000645 | 0.017077 ± 0.000645 | 0.086728 ± 0.001894 | 0.854629 ± 0.006473 | 0.168880 ± 0.010167 | 14.589526 ± 0.653513 |

## Seed-0 lambda sweep

| config | test global error | test improvement vs identity | test cosine mean | test cosine positive frac | test global alpha | test pred Δ norm med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lambda_cos=0.01 | 1.589500 | 0.017477 | 0.087518 | 0.851452 | 0.155716 | 15.848438 |
| lambda_cos=0.02 | 1.590096 | 0.016881 | 0.085968 | 0.848431 | 0.155694 | 15.617697 |
| lambda_cos=0.05 | 1.590619 | 0.016358 | 0.084597 | 0.848607 | 0.157234 | 15.298243 |
| lambda_cos=0.10 | 1.590938 | 0.016039 | 0.083834 | 0.846352 | 0.155209 | 15.481579 |

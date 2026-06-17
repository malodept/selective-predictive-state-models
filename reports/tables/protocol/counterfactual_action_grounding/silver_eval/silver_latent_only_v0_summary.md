# TartanAir silver latent-only v0 evaluation summary

This experiment evaluates existing V-JEPA 2.1 predictors on TartanAir silver matched-state groups mined with a latent-only nearest-neighbour protocol.

## Group mining result

The miner successfully produced enough groups:

| split | groups | candidates | same trajectory offdiag | same environment offdiag |
| --- | ---: | ---: | ---: | ---: |
| val | 8000 | 5 | 0.000000 | 1.000000 |
| test | 8000 | 5 | 0.000000 | 1.000000 |
| train | 20000 | 5 | 0.000000 | 0.954937 |

However, latent distances remain relatively large, so the groups should be treated as pseudo-counterfactual rather than exact same-state groups.

## Evaluation result

Chance top-1 for five candidates is 0.20.

| model | split | alpha | original top-1 | zero top-1 | shuffle top-1 | intervention gap original-zero | intervention gap original-shuffle |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V-JEPA MSE | val | 1.00 | 0.200500 | 0.200000 | 0.199825 | 0.000500 | 0.000675 |
| V-JEPA MSE | test | 1.00 | 0.200075 | 0.200000 | 0.200000 | 0.000075 | 0.000075 |
| V-JEPA candidate λ=0.2 | val | 1.00 | 0.200700 | 0.200000 | 0.200000 | 0.000700 | 0.000700 |
| V-JEPA candidate λ=0.2 | test | 1.00 | 0.200075 | 0.200000 | 0.200025 | 0.000075 | 0.000050 |

## Interpretation

The latent-only TartanAir silver groups do not produce a usable action-grounded identification benchmark. Both the MSE and candidate-objective V-JEPA models remain at chance on test.

This is a negative but informative result. It shows that simply mining neighbours in V-JEPA latent space is not sufficient to construct robust pseudo-counterfactual action groups.

## Decision

Do not train on `silver_latent_only_v0`.

Next steps should be:

1. either build a stricter pose-aware TartanAir silver miner using current pose and velocity;
2. or move directly to exact-intervention simulation with CALVIN / Habitat.

The current result strengthens the main diagnosis: the bottleneck is counterfactual identifiability, not merely encoder quality or loss design.

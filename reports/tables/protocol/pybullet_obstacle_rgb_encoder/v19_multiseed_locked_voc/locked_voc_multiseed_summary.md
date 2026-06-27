# SPSM v19 multi-cheap-seed locked VoC

This repeats the locked v15 routing comparison across cheap model seeds 0, 1, and 2.
The expensive model remains the same full seed0 predictor; the question is whether the routing conclusion is stable under cheap-model retraining.

## Mean utility on 3-block, H=72 across cheap seeds

| lambda | method | mean utility | std | mean route |
|---:|---|---:|---:|---:|
| 0.02 | `cheap` | 0.791768 | 0.059359 | 0.000000 |
| 0.02 | `full` | 0.798402 | 0.000000 | 1.000000 |
| 0.02 | `confidence` | 0.808458 | 0.031508 | 0.133979 |
| 0.02 | `context_cls` | 0.802534 | 0.033758 | 0.066990 |
| 0.02 | `knn_context` | 0.821065 | 0.024912 | 0.270379 |
| 0.02 | `rh_context` | 0.827022 | 0.029104 | 0.255044 |
| 0.02 | `knn_context_latent` | 0.823584 | 0.021734 | 0.184826 |
| 0.02 | `rh_context_latent` | 0.818483 | 0.026459 | 0.157385 |
| 0.02 | `oracle` | 0.893010 | 0.015985 | 0.103309 |
| 0.05 | `cheap` | 0.791768 | 0.059359 | 0.000000 |
| 0.05 | `full` | 0.768402 | 0.000000 | 1.000000 |
| 0.05 | `confidence` | 0.804439 | 0.033266 | 0.133979 |
| 0.05 | `context_cls` | 0.800525 | 0.035267 | 0.066990 |
| 0.05 | `knn_context` | 0.809685 | 0.033694 | 0.206618 |
| 0.05 | `rh_context` | 0.817272 | 0.036312 | 0.216303 |
| 0.05 | `knn_context_latent` | 0.818039 | 0.024111 | 0.184826 |
| 0.05 | `rh_context_latent` | 0.815456 | 0.026259 | 0.155771 |
| 0.05 | `oracle` | 0.889911 | 0.016461 | 0.103309 |
| 0.10 | `cheap` | 0.791768 | 0.059359 | 0.000000 |
| 0.10 | `full` | 0.718402 | 0.000000 | 1.000000 |
| 0.10 | `confidence` | 0.792575 | 0.038811 | 0.096852 |
| 0.10 | `context_cls` | 0.795400 | 0.040438 | 0.060533 |
| 0.10 | `knn_context` | 0.798144 | 0.039013 | 0.186441 |
| 0.10 | `rh_context` | 0.805085 | 0.048251 | 0.125101 |
| 0.10 | `knn_context_latent` | 0.808717 | 0.029614 | 0.112994 |
| 0.10 | `rh_context_latent` | 0.810331 | 0.031626 | 0.137207 |
| 0.10 | `oracle` | 0.884746 | 0.017566 | 0.103309 |
| 0.20 | `cheap` | 0.791768 | 0.059359 | 0.000000 |
| 0.20 | `full` | 0.618402 | 0.000000 | 1.000000 |
| 0.20 | `confidence` | 0.788216 | 0.054242 | 0.005650 |
| 0.20 | `context_cls` | 0.788378 | 0.044560 | 0.057304 |
| 0.20 | `knn_context` | 0.792090 | 0.061359 | 0.127522 |
| 0.20 | `rh_context` | 0.791122 | 0.060929 | 0.063761 |
| 0.20 | `knn_context_latent` | 0.796126 | 0.048332 | 0.054883 |
| 0.20 | `rh_context_latent` | 0.792897 | 0.054621 | 0.042776 |
| 0.20 | `oracle` | 0.874415 | 0.020695 | 0.103309 |
| 0.30 | `cheap` | 0.791768 | 0.059359 | 0.000000 |
| 0.30 | `full` | 0.518402 | 0.000000 | 1.000000 |
| 0.30 | `confidence` | 0.791768 | 0.059359 | 0.000000 |
| 0.30 | `context_cls` | 0.780630 | 0.051729 | 0.053269 |
| 0.30 | `knn_context` | 0.789750 | 0.062601 | 0.052462 |
| 0.30 | `rh_context` | 0.785311 | 0.069006 | 0.056497 |
| 0.30 | `knn_context_latent` | 0.790638 | 0.050240 | 0.054883 |
| 0.30 | `rh_context_latent` | 0.794431 | 0.058709 | 0.007264 |
| 0.30 | `oracle` | 0.864084 | 0.024645 | 0.103309 |

## Best fixed non-oracle method by cheap seed on 3-block, H=72

| lambda | cheap seed | best method | utility |
|---:|---:|---|---:|
| 0.02 | 0 | `knn_context` | 0.806538 |
| 0.02 | 1 | `rh_context` | 0.858886 |
| 0.02 | 2 | `rh_context` | 0.820339 |
| 0.05 | 0 | `knn_context_latent` | 0.797700 |
| 0.05 | 1 | `rh_context` | 0.857869 |
| 0.05 | 2 | `knn_context_latent` | 0.811743 |
| 0.10 | 0 | `rh_context_latent` | 0.781356 |
| 0.10 | 1 | `rh_context` | 0.856174 |
| 0.10 | 2 | `knn_context_latent` | 0.806295 |
| 0.20 | 0 | `knn_context_latent` | 0.745763 |
| 0.20 | 1 | `knn_context` | 0.851332 |
| 0.20 | 2 | `rh_context_latent` | 0.802421 |
| 0.30 | 0 | `knn_context_latent` | 0.737288 |
| 0.30 | 1 | `cheap` | 0.849879 |
| 0.30 | 2 | `rh_context_latent` | 0.800484 |

## Latent advantage over context-only on 3-block, H=72

| lambda | Δ KNN latent-context mean ± std | Δ RH latent-context mean ± std |
|---:|---:|---:|
| 0.02 | 0.002518 ± 0.006823 | -0.008539 ± 0.002844 |
| 0.05 | 0.008354 ± 0.009746 | -0.001816 ± 0.010068 |
| 0.10 | 0.010573 ± 0.009704 | 0.005246 ± 0.016639 |
| 0.20 | 0.004036 ± 0.013078 | 0.001776 ± 0.007828 |
| 0.30 | 0.000888 ± 0.012742 | 0.009120 ± 0.010331 |

## Interpretation

- If the latent advantage remains positive on average across cheap seeds, v15 is directionally stable.
- If the best method changes a lot across cheap seeds, the next scientific claim should emphasize cheap-model realization sensitivity.
- This result should be interpreted together with the v16 paired bootstrap over test queries.

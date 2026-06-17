# TartanAir silver latent-only filtered evaluation summary

This experiment tests whether the failure of `silver_latent_only_v0` is caused by weak group quality. We filter the mined matched-state groups by mean latent distance and evaluate only the closest 5%, 10%, and 20% of test groups.

## Filtered group quality

| split | keep frac | groups | latent mean | action mean | future mean | same trajectory offdiag | same environment offdiag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test | 0.05 | 400 | 0.508542 | 2.600079 | 24.694805 | 0.000000 | 1.000000 |
| test | 0.10 | 800 | 0.546853 | 2.472999 | 23.920416 | 0.000000 | 1.000000 |
| test | 0.20 | 1600 | 0.593240 | 2.555643 | 23.589785 | 0.000000 | 1.000000 |

Filtering produces closer latent groups while keeping different actions and different futures.

## V-JEPA MSE evaluation on filtered test groups

Chance top-1 is 0.20.

| keep frac | alpha | original top-1 | zero top-1 | shuffle top-1 | original-zero gap | original-shuffle gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 1.00 | 0.200000 | 0.200000 | 0.200500 | 0.000000 | -0.000500 |
| 0.10 | 1.00 | 0.200000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 |
| 0.20 | 1.00 | 0.200000 | 0.200000 | 0.199875 | 0.000000 | 0.000125 |

## Interpretation

Strict latent filtering does not reveal a hidden action-grounding signal. Even the closest latent groups remain at chance and show no reliable intervention gap.

This closes the `silver_latent_only_v0` direction. The failure is not only due to a few bad groups in the tail of the mined distribution. The latent-only neighbour criterion is insufficient for building pseudo-counterfactual action groups.

## Decision

Do not train on latent-only silver groups.

Next steps should move to either:

1. a pose-aware TartanAir silver miner using pose and velocity constraints;
2. exact-intervention branches in CALVIN or Habitat.

This result supports the broader diagnosis: action-grounded evaluation requires candidate groups where action is the principal varying factor. Latent-only nearest-neighbour mining does not guarantee that.

# TartanAir pose-aware silver v0 pilot summary

This pilot tests whether pose-aware matched-state groups improve the TartanAir pseudo-counterfactual protocol.

## Group quality

Compared with latent-only silver groups, pose-aware mining substantially improves physical state matching.

| protocol | test median position distance |
| --- | ---: |
| latent-only silver | 75.526161 |
| pose-aware silver v0 | 2.717982 |

Pose-aware groups are therefore much closer to the intended "near-same physical state" condition.

## V-JEPA MSE evaluation

Despite improved physical matching, the V-JEPA MSE predictor remains at chance.

| model | alpha | original top-1 | zero top-1 | shuffle top-1 |
| --- | ---: | ---: | ---: | ---: |
| V-JEPA MSE | 1.00 | 0.200000 | 0.200000 | 0.200000 |

## V-JEPA candidate-objective evaluation

The candidate-objective model also remains at chance.

| model | alpha | original top-1 | zero top-1 | shuffle top-1 |
| --- | ---: | ---: | ---: | ---: |
| V-JEPA candidate λ=0.2 | 1.00 | 0.200000 | 0.200000 | 0.200000 |

## Delta-transfer oracle

The delta-transfer oracle applies each candidate's true latent displacement to the anchor state:

`prediction_k = anchor_current + alpha * (candidate_future_k - candidate_current_k)`

| alpha | top-1 | mean rank | positive margin frac |
| ---: | ---: | ---: | ---: |
| 0.00 | 0.200000 | 3.000000 | 0.200000 |
| 1.00 | 0.205600 | 2.545200 | 0.204800 |

The oracle improves mean rank but barely improves top-1. This indicates that the pose-aware v0 groups contain some structure, but they are still not sufficiently identifiable as action-grounded counterfactual groups.

## Interpretation

Pose-aware matching solves the gross geometric mismatch of latent-only mining, but it does not yet create a strong action-grounded benchmark.

The bottleneck is not simply model failure. Even an optimistic delta-transfer oracle only weakly discriminates the correct future.

## Decision

Do not train on pose-aware v0 groups.

Run one stricter pose-aware oracle-only pilot. If the strict oracle does not show a clear top-1 signal, move to exact-intervention simulation such as CALVIN or Habitat.

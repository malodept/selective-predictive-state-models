# SPSM v5.3 OOD pilot summary — block3_h48_seed10

## Setup

- OOD shift: `num_blocked_directions=3`, `horizon=48`, `velocity=1.2`, `seed=10`.
- Dataset: 2000 exact-intervention simulator states, 5 actions per state, 10000 samples.
- Representation: frozen DINOv2 ViT-S/14 patch tokens, pooled from 16x16 to 4x4.
- Evaluation: v5.2 full state-action delta Transformer checkpoints, no retraining.

## Moving-only identifiable evaluation

| seed | strict top-1 | tie-aware top-1 | same-action/diff-state win | same-state/diff-action win |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.840796 | 0.840796 | 0.879353 | 0.992537 |
| 1 | 0.900498 | 0.900498 | 0.929104 | 0.992537 |
| 2 | 0.875622 | 0.875622 | 0.901741 | 0.993781 |

- Mean moving-only strict top-1: `0.872305 ± 0.029989`.
- Mean same-action/diff-state pairwise win: `0.903399 ± 0.024917`.
- Mean same-state/diff-action pairwise win: `0.992952 ± 0.000718`.

## Confidence and selective prediction

| seed | mean confidence | biased top-1 | strict top-1 | tie-aware top-1 | ECE tie-aware | ECE strict |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.799313 | 0.870000 | 0.676000 | 0.740667 | 0.066466 | 0.127211 |
| 1 | 0.800859 | 0.918000 | 0.724000 | 0.788667 | 0.018492 | 0.083159 |
| 2 | 0.807835 | 0.900000 | 0.704000 | 0.769333 | 0.041094 | 0.103835 |

## Risk-coverage summary

| coverage | mean strict top-1 among kept examples |
| ---: | ---: |
| 0.80 | 0.874167 |
| 0.60 | 0.975555 |
| 0.50 | 0.992000 |
| 0.40 | 0.996667 |

## Interpretation

The OOD shift creates a meaningful drop from the near-perfect v5.2 ID result while preserving strong action grounding. The largest weakness is same-action/different-state discrimination, not same-state/different-action discrimination. Confidence is useful for selective prediction: keeping the most confident 50–60% of examples recovers near-perfect strict top-1. This supports the v5.3 direction: reliability and value-of-computation should be evaluated under controlled exact-intervention shifts.

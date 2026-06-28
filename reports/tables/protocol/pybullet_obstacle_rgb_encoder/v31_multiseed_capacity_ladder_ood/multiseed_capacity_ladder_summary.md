# SPSM v31C multi-seed capacity ladder summary

This repeats the OOD capacity ladder across Tiny/Small/Medium/Full with seeds 0, 1, and 2.
The goal is to test whether the non-monotonic capacity result from v22/v29 is stable under retraining.

## Multi-seed OOD top-1 by capacity

| variant | Tiny | Small | Medium | Full | best mean | range of means |
|---|---:|---:|---:|---:|---|---:|
| 2-block, H=72 | 0.958±0.008 | 0.960±0.026 | 0.967±0.010 | 0.974±0.006 | `Full` | 0.016026 |
| 2-block, V=1.8 | 0.993±0.008 | 0.986±0.006 | 0.984±0.005 | 0.992±0.003 | `Tiny` | 0.009306 |
| 3-block, H=36 | 0.876±0.021 | 0.869±0.025 | 0.879±0.014 | 0.855±0.024 | `Medium` | 0.023635 |
| 3-block, H=48 | 0.853±0.020 | 0.867±0.017 | 0.884±0.001 | 0.872±0.030 | `Medium` | 0.030680 |
| 3-block, H=72 | 0.810±0.032 | 0.792±0.059 | 0.829±0.016 | 0.808±0.011 | `Medium` | 0.037127 |

## Best capacity per individual seed

| variant | seed0 | seed1 | seed2 |
|---|---|---|---|
| 2-block, H=72 | `Full` (0.975) | `Small` (0.975) | `Full` (0.980) |
| 2-block, V=1.8 | `Tiny` (0.995) | `Tiny` (1.000) | `Small` (0.992) |
| 3-block, H=36 | `Small` (0.897) | `Medium` (0.890) | `Medium` (0.883) |
| 3-block, H=48 | `Medium` (0.883) | `Full` (0.900) | `Medium` (0.883) |
| 3-block, H=72 | `Medium` (0.843) | `Small` (0.850) | `Tiny` (0.838) |

## State/action decomposition, mean over seeds

| variant | capacity | same-action/diff-state | same-state/diff-action |
|---|---|---:|---:|
| 2-block, H=72 | Full | 0.988±0.003 | 0.986±0.003 |
| 2-block, H=72 | Medium | 0.980±0.005 | 0.986±0.001 |
| 2-block, H=72 | Small | 0.972±0.017 | 0.987±0.002 |
| 2-block, H=72 | Tiny | 0.971±0.004 | 0.986±0.001 |
| 2-block, V=1.8 | Full | 0.997±0.003 | 0.999±0.000 |
| 2-block, V=1.8 | Medium | 0.990±0.003 | 0.997±0.001 |
| 2-block, V=1.8 | Small | 0.993±0.002 | 0.998±0.001 |
| 2-block, V=1.8 | Tiny | 0.998±0.004 | 0.998±0.001 |
| 3-block, H=36 | Full | 0.884±0.019 | 0.983±0.004 |
| 3-block, H=36 | Medium | 0.913±0.013 | 0.987±0.003 |
| 3-block, H=36 | Small | 0.916±0.019 | 0.988±0.008 |
| 3-block, H=36 | Tiny | 0.914±0.020 | 0.991±0.004 |
| 3-block, H=48 | Full | 0.903±0.025 | 0.993±0.001 |
| 3-block, H=48 | Medium | 0.915±0.002 | 0.995±0.001 |
| 3-block, H=48 | Small | 0.906±0.011 | 0.993±0.004 |
| 3-block, H=48 | Tiny | 0.895±0.021 | 0.989±0.004 |
| 3-block, H=72 | Full | 0.847±0.014 | 0.983±0.006 |
| 3-block, H=72 | Medium | 0.873±0.013 | 0.984±0.004 |
| 3-block, H=72 | Small | 0.843±0.054 | 0.991±0.001 |
| 3-block, H=72 | Tiny | 0.861±0.021 | 0.992±0.001 |

## Key robustness checks

- H=72 Medium - Full mean gap: 0.020985.
- H=72 Medium - Tiny mean gap: 0.019371.
- H=48 Medium - Full mean gap: 0.011609.
- H=48 Medium - Tiny mean gap: 0.030680.
- H=36 Small - Full mean gap: 0.013855.
- H=36 Small - Tiny mean gap: -0.007335.

## Interpretation guide

- If Medium remains above Full on H=72 and H=48 after averaging seeds, the non-monotonic capacity claim becomes much stronger.
- If the best capacity changes across seeds, the paper should emphasize capacity-seed sensitivity rather than a single deterministic ranking.
- If smaller capacities consistently beat Full on block3 variants, this supports the claim that larger predictors can be less reliable under constrained-geometry OOD.

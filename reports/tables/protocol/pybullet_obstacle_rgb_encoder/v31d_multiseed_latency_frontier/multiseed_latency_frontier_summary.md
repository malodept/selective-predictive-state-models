# SPSM v31D multi-seed latency-normalized capacity frontier

This repeats the latency-normalized fixed-capacity frontier using three independently trained seeds for every capacity.
Utility is `tie-aware top-1 - lambda * measured_relative_latency`, with Full batch-128 latency normalized to 1.0.

## Best mean fixed capacity by OOD variant and cost

| lambda | variant | best mean capacity | utility mean±std | top-1 mean±std | relative latency |
|---:|---|---|---:|---:|---:|
| 0.00 | 2-block, H=72 | `Full` | 0.974±0.006 | 0.974±0.006 | 1.000 |
| 0.00 | 2-block, V=1.8 | `Tiny` | 0.993±0.008 | 0.993±0.008 | 0.186 |
| 0.00 | 3-block, H=36 | `Medium` | 0.879±0.014 | 0.879±0.014 | 0.383 |
| 0.00 | 3-block, H=48 | `Medium` | 0.884±0.001 | 0.884±0.001 | 0.383 |
| 0.00 | 3-block, H=72 | `Medium` | 0.829±0.016 | 0.829±0.016 | 0.383 |
| 0.01 | 2-block, H=72 | `Full` | 0.964±0.006 | 0.974±0.006 | 1.000 |
| 0.01 | 2-block, V=1.8 | `Tiny` | 0.991±0.008 | 0.993±0.008 | 0.186 |
| 0.01 | 3-block, H=36 | `Medium` | 0.875±0.014 | 0.879±0.014 | 0.383 |
| 0.01 | 3-block, H=48 | `Medium` | 0.880±0.001 | 0.884±0.001 | 0.383 |
| 0.01 | 3-block, H=72 | `Medium` | 0.825±0.016 | 0.829±0.016 | 0.383 |
| 0.02 | 2-block, H=72 | `Medium` | 0.959±0.010 | 0.967±0.010 | 0.383 |
| 0.02 | 2-block, V=1.8 | `Tiny` | 0.990±0.008 | 0.993±0.008 | 0.186 |
| 0.02 | 3-block, H=36 | `Tiny` | 0.872±0.021 | 0.876±0.021 | 0.186 |
| 0.02 | 3-block, H=48 | `Medium` | 0.876±0.001 | 0.884±0.001 | 0.383 |
| 0.02 | 3-block, H=72 | `Medium` | 0.821±0.016 | 0.829±0.016 | 0.383 |
| 0.05 | 2-block, H=72 | `Small` | 0.949±0.026 | 0.960±0.026 | 0.201 |
| 0.05 | 2-block, V=1.8 | `Tiny` | 0.984±0.008 | 0.993±0.008 | 0.186 |
| 0.05 | 3-block, H=36 | `Tiny` | 0.867±0.021 | 0.876±0.021 | 0.186 |
| 0.05 | 3-block, H=48 | `Medium` | 0.865±0.001 | 0.884±0.001 | 0.383 |
| 0.05 | 3-block, H=72 | `Medium` | 0.810±0.016 | 0.829±0.016 | 0.383 |
| 0.10 | 2-block, H=72 | `Small` | 0.939±0.026 | 0.960±0.026 | 0.201 |
| 0.10 | 2-block, V=1.8 | `Tiny` | 0.975±0.008 | 0.993±0.008 | 0.186 |
| 0.10 | 3-block, H=36 | `Tiny` | 0.857±0.021 | 0.876±0.021 | 0.186 |
| 0.10 | 3-block, H=48 | `Small` | 0.847±0.017 | 0.867±0.017 | 0.201 |
| 0.10 | 3-block, H=72 | `Tiny` | 0.791±0.032 | 0.810±0.032 | 0.186 |
| 0.15 | 2-block, H=72 | `Tiny` | 0.930±0.008 | 0.958±0.008 | 0.186 |
| 0.15 | 2-block, V=1.8 | `Tiny` | 0.965±0.008 | 0.993±0.008 | 0.186 |
| 0.15 | 3-block, H=36 | `Tiny` | 0.848±0.021 | 0.876±0.021 | 0.186 |
| 0.15 | 3-block, H=48 | `Small` | 0.837±0.017 | 0.867±0.017 | 0.201 |
| 0.15 | 3-block, H=72 | `Tiny` | 0.782±0.032 | 0.810±0.032 | 0.186 |
| 0.20 | 2-block, H=72 | `Tiny` | 0.921±0.008 | 0.958±0.008 | 0.186 |
| 0.20 | 2-block, V=1.8 | `Tiny` | 0.956±0.008 | 0.993±0.008 | 0.186 |
| 0.20 | 3-block, H=36 | `Tiny` | 0.839±0.021 | 0.876±0.021 | 0.186 |
| 0.20 | 3-block, H=48 | `Small` | 0.827±0.017 | 0.867±0.017 | 0.201 |
| 0.20 | 3-block, H=72 | `Tiny` | 0.772±0.032 | 0.810±0.032 | 0.186 |
| 0.30 | 2-block, H=72 | `Tiny` | 0.902±0.008 | 0.958±0.008 | 0.186 |
| 0.30 | 2-block, V=1.8 | `Tiny` | 0.937±0.008 | 0.993±0.008 | 0.186 |
| 0.30 | 3-block, H=36 | `Tiny` | 0.820±0.021 | 0.876±0.021 | 0.186 |
| 0.30 | 3-block, H=48 | `Small` | 0.807±0.017 | 0.867±0.017 | 0.201 |
| 0.30 | 3-block, H=72 | `Tiny` | 0.754±0.032 | 0.810±0.032 | 0.186 |

## Hard OOD utility frontier at selected costs

| lambda | variant | Tiny | Small | Medium | Full | best |
|---:|---|---:|---:|---:|---:|---|
| 0.00 | 3-block, H=36 | 0.876±0.021 | 0.869±0.025 | 0.879±0.014 | 0.855±0.024 | `Medium` |
| 0.00 | 3-block, H=48 | 0.853±0.020 | 0.867±0.017 | 0.884±0.001 | 0.872±0.030 | `Medium` |
| 0.00 | 3-block, H=72 | 0.810±0.032 | 0.792±0.059 | 0.829±0.016 | 0.808±0.011 | `Medium` |
| 0.05 | 3-block, H=36 | 0.867±0.021 | 0.859±0.025 | 0.859±0.014 | 0.805±0.024 | `Tiny` |
| 0.05 | 3-block, H=48 | 0.844±0.020 | 0.857±0.017 | 0.865±0.001 | 0.822±0.030 | `Medium` |
| 0.05 | 3-block, H=72 | 0.800±0.032 | 0.782±0.059 | 0.810±0.016 | 0.758±0.011 | `Medium` |
| 0.10 | 3-block, H=36 | 0.857±0.021 | 0.849±0.025 | 0.840±0.014 | 0.755±0.024 | `Tiny` |
| 0.10 | 3-block, H=48 | 0.835±0.020 | 0.847±0.017 | 0.846±0.001 | 0.772±0.030 | `Small` |
| 0.10 | 3-block, H=72 | 0.791±0.032 | 0.772±0.059 | 0.791±0.016 | 0.708±0.011 | `Tiny` |
| 0.20 | 3-block, H=36 | 0.839±0.021 | 0.829±0.025 | 0.802±0.014 | 0.655±0.024 | `Tiny` |
| 0.20 | 3-block, H=48 | 0.816±0.020 | 0.827±0.017 | 0.807±0.001 | 0.672±0.030 | `Small` |
| 0.20 | 3-block, H=72 | 0.772±0.032 | 0.752±0.059 | 0.752±0.016 | 0.608±0.011 | `Tiny` |

## Crossover diagnostics on hard OOD

| variant | Medium-Full top-1 gap | Medium-Tiny top-1 gap | Tiny-Medium latency advantage | approx Medium-vs-Tiny λ crossover |
|---|---:|---:|---:|---:|
| 3-block, H=36 | 0.023635 | 0.002445 | 0.196325 | 0.0125 |
| 3-block, H=48 | 0.011609 | 0.030680 | 0.196325 | 0.1563 |
| 3-block, H=72 | 0.020985 | 0.019371 | 0.196325 | 0.0987 |

## Interpretation

- The multi-seed result supports the non-monotonic capacity claim: Full is not the best mean model on any block3 hard OOD shift.
- Medium is the best mean accuracy model on H=36, H=48, and H=72, but the best individual seed can vary.
- At higher compute cost, Tiny can become optimal because Medium's accuracy advantage no longer pays for its additional measured latency.
- This is a stronger and more defensible version of the v25/v29 capacity frontier.

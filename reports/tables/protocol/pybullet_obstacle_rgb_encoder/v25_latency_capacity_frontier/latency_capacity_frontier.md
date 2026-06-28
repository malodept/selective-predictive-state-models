# SPSM v25 latency-normalized capacity frontier

This rebuilds the v23 fixed-capacity frontier using measured batch-128 forward latency instead of checkpoint size.
The objective is `utility = top1 - lambda * relative_latency`, where Full latency is normalized to 1.0.

## Measured latency-normalized best fixed capacity

| lambda | variant | best model | utility | top-1 | relative latency | latency ms @128 |
|---:|---|---|---:|---:|---:|---:|
| 0.00 | 2-block, H=72 | `Full` | 0.974760 | 0.974760 | 1.000000 | 3.4437 |
| 0.00 | 2-block, V=1.8 | `Tiny` | 0.994924 | 0.994924 | 0.186312 | 0.6416 |
| 0.00 | 3-block, H=36 | `Small` | 0.897311 | 0.897311 | 0.201019 | 0.6922 |
| 0.00 | 3-block, H=48 | `Medium` | 0.883085 | 0.883085 | 0.382637 | 1.3177 |
| 0.00 | 3-block, H=72 | `Medium` | 0.842615 | 0.842615 | 0.382637 | 1.3177 |
| 0.01 | 2-block, H=72 | `Full` | 0.964760 | 0.974760 | 1.000000 | 3.4437 |
| 0.01 | 2-block, V=1.8 | `Tiny` | 0.993061 | 0.994924 | 0.186312 | 0.6416 |
| 0.01 | 3-block, H=36 | `Small` | 0.895301 | 0.897311 | 0.201019 | 0.6922 |
| 0.01 | 3-block, H=48 | `Medium` | 0.879259 | 0.883085 | 0.382637 | 1.3177 |
| 0.01 | 3-block, H=72 | `Medium` | 0.838789 | 0.842615 | 0.382637 | 1.3177 |
| 0.02 | 2-block, H=72 | `Full` | 0.954760 | 0.974760 | 1.000000 | 3.4437 |
| 0.02 | 2-block, V=1.8 | `Tiny` | 0.991198 | 0.994924 | 0.186312 | 0.6416 |
| 0.02 | 3-block, H=36 | `Small` | 0.893291 | 0.897311 | 0.201019 | 0.6922 |
| 0.02 | 3-block, H=48 | `Medium` | 0.875432 | 0.883085 | 0.382637 | 1.3177 |
| 0.02 | 3-block, H=72 | `Medium` | 0.834962 | 0.842615 | 0.382637 | 1.3177 |
| 0.05 | 2-block, H=72 | `Tiny` | 0.943809 | 0.953125 | 0.186312 | 0.6416 |
| 0.05 | 2-block, V=1.8 | `Tiny` | 0.985608 | 0.994924 | 0.186312 | 0.6416 |
| 0.05 | 3-block, H=36 | `Small` | 0.887260 | 0.897311 | 0.201019 | 0.6922 |
| 0.05 | 3-block, H=48 | `Medium` | 0.863953 | 0.883085 | 0.382637 | 1.3177 |
| 0.05 | 3-block, H=72 | `Medium` | 0.823483 | 0.842615 | 0.382637 | 1.3177 |
| 0.10 | 2-block, H=72 | `Tiny` | 0.934494 | 0.953125 | 0.186312 | 0.6416 |
| 0.10 | 2-block, V=1.8 | `Tiny` | 0.976293 | 0.994924 | 0.186312 | 0.6416 |
| 0.10 | 3-block, H=36 | `Small` | 0.877209 | 0.897311 | 0.201019 | 0.6922 |
| 0.10 | 3-block, H=48 | `Tiny` | 0.849528 | 0.868159 | 0.186312 | 0.6416 |
| 0.10 | 3-block, H=72 | `Medium` | 0.804351 | 0.842615 | 0.382637 | 1.3177 |
| 0.15 | 2-block, H=72 | `Tiny` | 0.925178 | 0.953125 | 0.186312 | 0.6416 |
| 0.15 | 2-block, V=1.8 | `Tiny` | 0.966977 | 0.994924 | 0.186312 | 0.6416 |
| 0.15 | 3-block, H=36 | `Small` | 0.867158 | 0.897311 | 0.201019 | 0.6922 |
| 0.15 | 3-block, H=48 | `Tiny` | 0.840212 | 0.868159 | 0.186312 | 0.6416 |
| 0.15 | 3-block, H=72 | `Tiny` | 0.788034 | 0.815981 | 0.186312 | 0.6416 |
| 0.20 | 2-block, H=72 | `Tiny` | 0.915863 | 0.953125 | 0.186312 | 0.6416 |
| 0.20 | 2-block, V=1.8 | `Tiny` | 0.957662 | 0.994924 | 0.186312 | 0.6416 |
| 0.20 | 3-block, H=36 | `Tiny` | 0.857604 | 0.894866 | 0.186312 | 0.6416 |
| 0.20 | 3-block, H=48 | `Tiny` | 0.830897 | 0.868159 | 0.186312 | 0.6416 |
| 0.20 | 3-block, H=72 | `Tiny` | 0.778719 | 0.815981 | 0.186312 | 0.6416 |
| 0.30 | 2-block, H=72 | `Tiny` | 0.897231 | 0.953125 | 0.186312 | 0.6416 |
| 0.30 | 2-block, V=1.8 | `Tiny` | 0.939030 | 0.994924 | 0.186312 | 0.6416 |
| 0.30 | 3-block, H=36 | `Tiny` | 0.838972 | 0.894866 | 0.186312 | 0.6416 |
| 0.30 | 3-block, H=48 | `Tiny` | 0.812265 | 0.868159 | 0.186312 | 0.6416 |
| 0.30 | 3-block, H=72 | `Tiny` | 0.760087 | 0.815981 | 0.186312 | 0.6416 |

## Accuracy-latency table

| variant | model | top-1 | latency ms @128 | relative latency | params |
|---|---|---:|---:|---:|---:|
| 2-block, H=72 | Tiny | 0.953125 | 0.6416 | 0.186312 | 100928 |
| 2-block, H=72 | Small | 0.929087 | 0.6922 | 0.201019 | 299776 |
| 2-block, H=72 | Medium | 0.955529 | 1.3177 | 0.382637 | 1041792 |
| 2-block, H=72 | Full | 0.974760 | 3.4437 | 1.000000 | 5627136 |
| 2-block, V=1.8 | Tiny | 0.994924 | 0.6416 | 0.186312 | 100928 |
| 2-block, V=1.8 | Small | 0.987310 | 0.6922 | 0.201019 | 299776 |
| 2-block, V=1.8 | Medium | 0.979695 | 1.3177 | 0.382637 | 1041792 |
| 2-block, V=1.8 | Full | 0.994924 | 3.4437 | 1.000000 | 5627136 |
| 3-block, H=36 | Tiny | 0.894866 | 0.6416 | 0.186312 | 100928 |
| 3-block, H=36 | Small | 0.897311 | 0.6922 | 0.201019 | 299776 |
| 3-block, H=36 | Medium | 0.863081 | 1.3177 | 0.382637 | 1041792 |
| 3-block, H=36 | Full | 0.828851 | 3.4437 | 1.000000 | 5627136 |
| 3-block, H=48 | Tiny | 0.868159 | 0.6416 | 0.186312 | 100928 |
| 3-block, H=48 | Small | 0.853234 | 0.6922 | 0.201019 | 299776 |
| 3-block, H=48 | Medium | 0.883085 | 1.3177 | 0.382637 | 1041792 |
| 3-block, H=48 | Full | 0.840796 | 3.4437 | 1.000000 | 5627136 |
| 3-block, H=72 | Tiny | 0.815981 | 0.6416 | 0.186312 | 100928 |
| 3-block, H=72 | Small | 0.731235 | 0.6922 | 0.201019 | 299776 |
| 3-block, H=72 | Medium | 0.842615 | 1.3177 | 0.382637 | 1041792 |
| 3-block, H=72 | Full | 0.818402 | 3.4437 | 1.000000 | 5627136 |

## Key crossovers

| variant | comparison | lambda crossover | interpretation |
|---|---|---:|---|
| 2-block, H=72 | Full vs Tiny | 0.0266 | Full preferred below crossover, Tiny above |
| 2-block, H=72 | Medium vs Tiny | 0.0122 | Medium preferred below crossover, Tiny above |
| 2-block, H=72 | Medium vs Small | 0.1456 | Medium preferred below crossover, Small above |
| 2-block, H=72 | Small vs Tiny | -1.6345 | higher-cost model is not justified by accuracy |
| 2-block, V=1.8 | Full vs Tiny | 0.0000 | higher-cost model is not justified by accuracy |
| 2-block, V=1.8 | Medium vs Tiny | -0.0776 | higher-cost model is not justified by accuracy |
| 2-block, V=1.8 | Medium vs Small | -0.0419 | higher-cost model is not justified by accuracy |
| 2-block, V=1.8 | Small vs Tiny | -0.5177 | higher-cost model is not justified by accuracy |
| 3-block, H=36 | Full vs Tiny | -0.0811 | higher-cost model is not justified by accuracy |
| 3-block, H=36 | Medium vs Tiny | -0.1619 | higher-cost model is not justified by accuracy |
| 3-block, H=36 | Medium vs Small | -0.1885 | higher-cost model is not justified by accuracy |
| 3-block, H=36 | Small vs Tiny | 0.1663 | Small preferred below crossover, Tiny above |
| 3-block, H=48 | Full vs Tiny | -0.0336 | higher-cost model is not justified by accuracy |
| 3-block, H=48 | Medium vs Tiny | 0.0760 | Medium preferred below crossover, Tiny above |
| 3-block, H=48 | Medium vs Small | 0.1644 | Medium preferred below crossover, Small above |
| 3-block, H=48 | Small vs Tiny | -1.0149 | higher-cost model is not justified by accuracy |
| 3-block, H=72 | Full vs Tiny | 0.0030 | Full preferred below crossover, Tiny above |
| 3-block, H=72 | Medium vs Tiny | 0.1357 | Medium preferred below crossover, Tiny above |
| 3-block, H=72 | Medium vs Small | 0.6133 | Medium preferred below crossover, Small above |
| 3-block, H=72 | Small vs Tiny | -5.7625 | higher-cost model is not justified by accuracy |

## Interpretation

- Measured latency makes Tiny/Small less cheap than checkpoint size suggested, so v23 overstated the cost advantage of Tiny.
- Capacity remains non-monotonic: Medium is the best accuracy model on the hardest 3-block H=72 shift, while Full is not consistently best.
- For low compute cost, Medium remains valuable on hard long-horizon OOD; for high cost, Tiny becomes optimal.
- This supports a multi-capacity value-of-computation formulation: choosing among Tiny, Small, Medium, and Full is more faithful than binary Small-vs-Full routing.

# SPSM v23 capacity cost frontier

This is an aggregate fixed-model frontier: choose one capacity for all examples in a variant.
Cost is currently approximated by checkpoint size relative to Full. This is a proxy; measured latency should replace it later.

## Best fixed capacity by OOD variant and cost

| lambda | variant | best model | utility | top-1 | relative cost |
|---:|---|---|---:|---:|---:|
| 0.00 | 2-block, H=72 | `Full` | 0.974760 | 0.974760 | 1.000000 |
| 0.00 | 2-block, V=1.8 | `Tiny` | 0.994924 | 0.994924 | 0.018304 |
| 0.00 | 3-block, H=36 | `Small` | 0.897311 | 0.897311 | 0.053614 |
| 0.00 | 3-block, H=48 | `Medium` | 0.883085 | 0.883085 | 0.185572 |
| 0.00 | 3-block, H=72 | `Medium` | 0.842615 | 0.842615 | 0.185572 |
| 0.01 | 2-block, H=72 | `Full` | 0.964760 | 0.974760 | 1.000000 |
| 0.01 | 2-block, V=1.8 | `Tiny` | 0.994741 | 0.994924 | 0.018304 |
| 0.01 | 3-block, H=36 | `Small` | 0.896775 | 0.897311 | 0.053614 |
| 0.01 | 3-block, H=48 | `Medium` | 0.881229 | 0.883085 | 0.185572 |
| 0.01 | 3-block, H=72 | `Medium` | 0.840759 | 0.842615 | 0.185572 |
| 0.02 | 2-block, H=72 | `Full` | 0.954760 | 0.974760 | 1.000000 |
| 0.02 | 2-block, V=1.8 | `Tiny` | 0.994558 | 0.994924 | 0.018304 |
| 0.02 | 3-block, H=36 | `Small` | 0.896239 | 0.897311 | 0.053614 |
| 0.02 | 3-block, H=48 | `Medium` | 0.879374 | 0.883085 | 0.185572 |
| 0.02 | 3-block, H=72 | `Medium` | 0.838904 | 0.842615 | 0.185572 |
| 0.05 | 2-block, H=72 | `Tiny` | 0.952210 | 0.953125 | 0.018304 |
| 0.05 | 2-block, V=1.8 | `Tiny` | 0.994009 | 0.994924 | 0.018304 |
| 0.05 | 3-block, H=36 | `Small` | 0.894630 | 0.897311 | 0.053614 |
| 0.05 | 3-block, H=48 | `Medium` | 0.873806 | 0.883085 | 0.185572 |
| 0.05 | 3-block, H=72 | `Medium` | 0.833336 | 0.842615 | 0.185572 |
| 0.10 | 2-block, H=72 | `Tiny` | 0.951295 | 0.953125 | 0.018304 |
| 0.10 | 2-block, V=1.8 | `Tiny` | 0.993094 | 0.994924 | 0.018304 |
| 0.10 | 3-block, H=36 | `Tiny` | 0.893036 | 0.894866 | 0.018304 |
| 0.10 | 3-block, H=48 | `Tiny` | 0.866329 | 0.868159 | 0.018304 |
| 0.10 | 3-block, H=72 | `Medium` | 0.824058 | 0.842615 | 0.185572 |
| 0.20 | 2-block, H=72 | `Tiny` | 0.949464 | 0.953125 | 0.018304 |
| 0.20 | 2-block, V=1.8 | `Tiny` | 0.991263 | 0.994924 | 0.018304 |
| 0.20 | 3-block, H=36 | `Tiny` | 0.891205 | 0.894866 | 0.018304 |
| 0.20 | 3-block, H=48 | `Tiny` | 0.864498 | 0.868159 | 0.018304 |
| 0.20 | 3-block, H=72 | `Tiny` | 0.812320 | 0.815981 | 0.018304 |
| 0.30 | 2-block, H=72 | `Tiny` | 0.947634 | 0.953125 | 0.018304 |
| 0.30 | 2-block, V=1.8 | `Tiny` | 0.989433 | 0.994924 | 0.018304 |
| 0.30 | 3-block, H=36 | `Tiny` | 0.889375 | 0.894866 | 0.018304 |
| 0.30 | 3-block, H=48 | `Tiny` | 0.862668 | 0.868159 | 0.018304 |
| 0.30 | 3-block, H=72 | `Tiny` | 0.810490 | 0.815981 | 0.018304 |

## Accuracy-cost table

| variant | model | top-1 | checkpoint MB | relative cost | same-action/diff-state | same-state/diff-action |
|---|---|---:|---:|---:|---:|---:|
| 2-block, H=72 | Tiny | 0.953125 | 0.39 | 0.018304 | 0.968750 | 0.986779 |
| 2-block, H=72 | Small | 0.929087 | 1.15 | 0.053614 | 0.953125 | 0.985577 |
| 2-block, H=72 | Medium | 0.955529 | 3.99 | 0.185572 | 0.974760 | 0.986779 |
| 2-block, H=72 | Full | 0.974760 | 21.48 | 1.000000 | 0.986779 | 0.986779 |
| 2-block, V=1.8 | Tiny | 0.994924 | 0.39 | 0.018304 | 1.000000 | 0.997462 |
| 2-block, V=1.8 | Small | 0.987310 | 1.15 | 0.053614 | 0.992386 | 0.998731 |
| 2-block, V=1.8 | Medium | 0.979695 | 3.99 | 0.185572 | 0.988579 | 0.997462 |
| 2-block, V=1.8 | Full | 0.994924 | 21.48 | 1.000000 | 0.998731 | 0.998731 |
| 3-block, H=36 | Tiny | 0.894866 | 0.39 | 0.018304 | 0.933985 | 0.991443 |
| 3-block, H=36 | Small | 0.897311 | 1.15 | 0.053614 | 0.936430 | 0.995110 |
| 3-block, H=36 | Medium | 0.863081 | 3.99 | 0.185572 | 0.902200 | 0.984108 |
| 3-block, H=36 | Full | 0.828851 | 21.48 | 1.000000 | 0.863081 | 0.979218 |
| 3-block, H=48 | Tiny | 0.868159 | 0.39 | 0.018304 | 0.909204 | 0.992537 |
| 3-block, H=48 | Small | 0.853234 | 1.15 | 0.053614 | 0.896766 | 0.997512 |
| 3-block, H=48 | Medium | 0.883085 | 3.99 | 0.185572 | 0.916667 | 0.995025 |
| 3-block, H=48 | Full | 0.840796 | 21.48 | 1.000000 | 0.879353 | 0.992537 |
| 3-block, H=72 | Tiny | 0.815981 | 0.39 | 0.018304 | 0.868039 | 0.992736 |
| 3-block, H=72 | Small | 0.731235 | 1.15 | 0.053614 | 0.786925 | 0.991525 |
| 3-block, H=72 | Medium | 0.842615 | 3.99 | 0.185572 | 0.887409 | 0.979419 |
| 3-block, H=72 | Full | 0.818402 | 21.48 | 1.000000 | 0.857143 | 0.976998 |

## Interpretation

- Capacity is not monotonic: Medium is best on the hardest 3-block H=72 shift, while Full is not consistently best.
- Because Full has much higher relative cost, any positive compute penalty makes Medium/Tiny/Small more attractive unless Full has a large accuracy advantage.
- This supports replacing binary small-to-full routing with a multi-capacity value-of-computation problem.
- The next rigorous step is to measure actual inference latency and then build per-instance routing across capacities.

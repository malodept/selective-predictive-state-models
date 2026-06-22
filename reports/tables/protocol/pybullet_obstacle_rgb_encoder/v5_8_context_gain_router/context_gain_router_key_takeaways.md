# SPSM v5.8 — Key takeaways

## Main result

Adding simple observable context features (`horizon_norm`, `velocity_norm`) substantially improves the small-full to full value-of-computation router.

The route decision is no longer based only on cheap-model confidence. It can adapt to the environment context.

## Important hard-OOD cases

### `block3_h36_seed13`

In this variant, `small_full` is stronger than `full_seed0`.

At `lambda = 0.10`:

| method | utility | route rate |
|---|---:|---:|
| cheap only | 0.897311 | 0.000000 |
| full only | 0.728851 | 1.000000 |
| confidence threshold | 0.846455 | 0.215159 |
| context-aware router | 0.891198 | 0.012225 |
| oracle | 0.936919 | 0.044010 |

Interpretation: the context-aware router correctly avoids most harmful routing to the expensive model.

### `block3_h72_seed14`

In this variant, `full_seed0` helps more often, especially at low and moderate compute cost.

At `lambda = 0.10`:

| method | utility | route rate |
|---|---:|---:|
| cheap only | 0.731235 | 0.000000 |
| full only | 0.718402 | 1.000000 |
| confidence threshold | 0.755690 | 0.288136 |
| context-aware router | 0.773123 | 0.259080 |
| oracle | 0.879419 | 0.164649 |

Interpretation: the context-aware router improves over cheap-only, full-only, and confidence-threshold routing.

## Scientific interpretation

The v5.8 result supports the central SPSM value-of-computation claim:

A predictive-state model should not always pay for expensive prediction. It should estimate when the additional computation is likely to improve decision quality.

The important qualitative result is that the expensive model is not uniformly better. In some OOD regimes, the smaller model is safer; in others, the larger model provides useful corrections. This makes value-of-computation routing a real problem rather than a trivial confidence threshold.

## Current limitation

The context-aware router remains far from oracle performance, especially on hard OOD. This suggests that cheap-model confidence and simple environment metadata are not sufficient. Future routers should include explicit geometry or OOD-complexity features.

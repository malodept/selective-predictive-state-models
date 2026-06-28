# Paper assets v31 multi-seed capacity

## Main update

The capacity frontier is now validated across three independently trained seeds for every capacity.
The single-seed v29 claim is strengthened: Full is not the best mean model on any block3 hard-OOD shift.

## Recommended replacement claim

Under constrained-geometry OOD, predictive capacity is not monotonically ordered.
Medium-capacity predictors outperform the largest predictor on average across hard block3 shifts, while measured latency determines whether Medium, Small, or Tiny is optimal at a given compute cost.
This makes value-of-computation a multi-capacity selection problem rather than a simple small-to-large cascade.

## Key numbers

- 3-block H=36: Medium 0.879±0.014 vs Full 0.855±0.024.
- 3-block H=48: Medium 0.884±0.001 vs Full 0.872±0.030.
- 3-block H=72: Medium 0.829±0.016 vs Full 0.808±0.011.
- H=72 Medium-vs-Tiny latency crossover: λ≈0.099.
- H=48 Medium-vs-Tiny latency crossover: λ≈0.156.

## Claim boundary

Do not claim that Medium is always best seed-by-seed.
The correct claim is that capacity is non-monotonic in expectation over retraining seeds, and that the best capacity is shift-, seed-, and cost-dependent.

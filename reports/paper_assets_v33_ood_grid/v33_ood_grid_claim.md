# Paper assets v33 OOD grid

## Main update

The OOD result now uses a systematic mini-grid rather than a small set of hand-picked variants:
blocked directions {1,2,3,4}, horizons {36,72}, and three layout seeds.

## Recommended claim

Controlled OOD geometry induces non-linear reliability regimes.
Capacity is not monotonically ordered across the grid: Full is best mainly in the easy blocked=2 regime, Medium dominates several intermediate constrained-geometry regimes, and smaller capacities become competitive or best in the extreme blocked=4 regime.

## Key numbers

- H=36, blocked=3: Medium 0.927 vs Full 0.869.
- H=72, blocked=1: Medium 0.927 vs Full 0.879.
- H=72, blocked=3: Medium 0.839 vs Full 0.813.
- H=72, blocked=4: Small 0.602 vs Full 0.543.
- blocked=4 gives nearly identical H=36/H=72 ranking behavior. Sanity checks show that future physical positions are nearly identical, while rendered RGB and DINOv2 future features are not bitwise identical; this should be framed as an extreme constrained-geometry regime where horizon has little physical effect, not as two fully independent regimes.

## Claim boundary

Do not claim monotonic degradation with blocked count. The grid shows non-linear geometry regimes, not a simple scalar difficulty axis. Also avoid over-counting block4 H=36 and H=72 as independent evidence, since their physical endpoints are almost identical.
Do not claim Medium is universally best. The correct claim is that capacity ordering is shift-dependent and non-monotonic.

# v47B compute-regime-map claim

## Main idea

The compute-regime map turns value-of-computation into a query-level diagnostic. Instead of asking only which capacity has the best average utility, it decomposes each OOD regime into all-correct, none-correct, useful-compute, anti-rescue, and other mixed cases.

## Correct scientific claim

Additional predictive compute is not a monotone fallback. It can be unnecessary when all capacities are correct, useful when Tiny fails but another capacity succeeds, insufficient when all capacities fail, or harmful when Tiny is correct but Full is wrong. This is the interventional value-of-computation view.

## Result pattern

H=36 is mostly all-correct with limited useful-compute mass. H=48 has more useful-compute mass and the strongest stable margin-gated routing gains. H=72 has the largest useful-compute and anti-rescue rates, plus the largest oracle headroom, explaining why it remains difficult: value exists, but monotone deferral is unsafe and current routing recovers only part of the oracle.

## Paper placement

This should become a central conceptual table/figure in the routing section. It explains why non-monotonic capacity and partial routing gains arise, instead of merely reporting them.

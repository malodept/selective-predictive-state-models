# SPSM v32B seed-aligned multi-capacity routing

This evaluates multi-capacity routing across three seed-aligned ladders.
Each ladder contains Tiny/Small/Medium/Full with the same training seed; results are averaged across ladder seeds.
The router is trained on all non-heldout variants and uses only Tiny diagnostics, action, and context features.

## Best learned router vs best fixed and oracle, averaged over ladder seeds

| lambda | heldout | best fixed | fixed util | best learned | learned util | oracle util | learned-fixed | oracle-learned | selection Tiny/Small/Medium/Full |
|---:|---|---|---:|---|---:|---:|---:|---:|---|
| 0.00 | 3-block, H=36 | `Medium` | 0.878566 | `logreg_oracle_label` | 0.892421 | 0.947025 | 0.013855 | 0.054605 | 0.74/0.06/0.01/0.18 |
| 0.00 | 3-block, H=48 | `Medium` | 0.883914 | `ridge_expected_utility` | 0.910448 | 0.949420 | 0.026534 | 0.038972 | 0.73/0.02/0.23/0.02 |
| 0.00 | 3-block, H=72 | `Medium` | 0.828894 | `rf_expected_utility` | 0.836158 | 0.935432 | 0.007264 | 0.099274 | 0.35/0.30/0.21/0.14 |
| 0.01 | 3-block, H=36 | `Medium` | 0.874739 | `logreg_oracle_label` | 0.889028 | 0.945078 | 0.014289 | 0.056050 | 0.74/0.06/0.01/0.18 |
| 0.01 | 3-block, H=48 | `Medium` | 0.880087 | `ridge_expected_utility` | 0.908903 | 0.947377 | 0.028816 | 0.038474 | 0.74/0.02/0.24/0.00 |
| 0.01 | 3-block, H=72 | `Medium` | 0.825068 | `rf_expected_utility` | 0.835315 | 0.933361 | 0.010247 | 0.098046 | 0.59/0.08/0.20/0.12 |
| 0.02 | 3-block, H=36 | `Tiny` | 0.872394 | `logreg_oracle_label` | 0.885635 | 0.943131 | 0.013241 | 0.057496 | 0.74/0.06/0.01/0.18 |
| 0.02 | 3-block, H=48 | `Medium` | 0.876261 | `ridge_expected_utility` | 0.906622 | 0.945334 | 0.030361 | 0.038712 | 0.74/0.02/0.23/0.00 |
| 0.02 | 3-block, H=72 | `Medium` | 0.821242 | `rf_expected_utility` | 0.832300 | 0.931290 | 0.011059 | 0.098990 | 0.61/0.08/0.21/0.11 |
| 0.05 | 3-block, H=36 | `Tiny` | 0.866805 | `ridge_expected_utility` | 0.878397 | 0.937290 | 0.011592 | 0.058893 | 0.35/0.37/0.28/0.00 |
| 0.05 | 3-block, H=48 | `Medium` | 0.864782 | `ridge_expected_utility` | 0.899809 | 0.939207 | 0.035027 | 0.039398 | 0.76/0.03/0.22/0.00 |
| 0.05 | 3-block, H=72 | `Medium` | 0.809762 | `rf_expected_utility` | 0.827699 | 0.925078 | 0.017937 | 0.097379 | 0.63/0.09/0.22/0.06 |
| 0.10 | 3-block, H=36 | `Tiny` | 0.857489 | `ridge_expected_utility` | 0.869008 | 0.927556 | 0.011518 | 0.058548 | 0.37/0.37/0.26/0.00 |
| 0.10 | 3-block, H=48 | `Small` | 0.847228 | `ridge_expected_utility` | 0.891170 | 0.928994 | 0.043942 | 0.037824 | 0.77/0.03/0.20/0.00 |
| 0.10 | 3-block, H=72 | `Tiny` | 0.790893 | `rf_expected_utility` | 0.813395 | 0.914725 | 0.022502 | 0.101330 | 0.65/0.11/0.22/0.03 |
| 0.15 | 3-block, H=36 | `Tiny` | 0.848174 | `ridge_expected_utility` | 0.855404 | 0.917821 | 0.007230 | 0.062417 | 0.40/0.38/0.22/0.00 |
| 0.15 | 3-block, H=48 | `Small` | 0.837177 | `ridge_expected_utility` | 0.883655 | 0.918781 | 0.046478 | 0.035126 | 0.79/0.03/0.18/0.00 |
| 0.15 | 3-block, H=72 | `Tiny` | 0.781577 | `rf_expected_utility` | 0.803155 | 0.904371 | 0.021578 | 0.101216 | 0.66/0.12/0.21/0.01 |
| 0.20 | 3-block, H=36 | `Tiny` | 0.838858 | `rf_expected_utility` | 0.846207 | 0.908086 | 0.007348 | 0.061879 | 0.73/0.10/0.17/0.00 |
| 0.20 | 3-block, H=48 | `Small` | 0.827126 | `ridge_expected_utility` | 0.873157 | 0.908567 | 0.046030 | 0.035411 | 0.80/0.03/0.17/0.00 |
| 0.20 | 3-block, H=72 | `Tiny` | 0.772261 | `rf_expected_utility` | 0.792442 | 0.894017 | 0.020181 | 0.101575 | 0.68/0.13/0.19/0.00 |
| 0.30 | 3-block, H=36 | `Tiny` | 0.820227 | `rf_expected_utility` | 0.828282 | 0.888616 | 0.008055 | 0.060334 | 0.75/0.11/0.15/0.00 |
| 0.30 | 3-block, H=48 | `Small` | 0.807024 | `ridge_expected_utility` | 0.853726 | 0.888141 | 0.046701 | 0.034416 | 0.82/0.05/0.12/0.00 |
| 0.30 | 3-block, H=72 | `Tiny` | 0.753630 | `rf_expected_utility` | 0.772594 | 0.873310 | 0.018964 | 0.100716 | 0.70/0.15/0.15/0.00 |

## Per-seed best learned gains

| lambda | heldout | seed | best fixed | best learned | learned-fixed |
|---:|---|---:|---|---|---:|
| 0.00 | 3-block, H=36 | 0 | `Small` | `rf_oracle_label` | -0.004890 |
| 0.00 | 3-block, H=36 | 1 | `Medium` | `logreg_oracle_label` | 0.004890 |
| 0.00 | 3-block, H=36 | 2 | `Medium` | `logreg_oracle_label` | 0.012225 |
| 0.00 | 3-block, H=48 | 0 | `Medium` | `ridge_expected_utility` | 0.039801 |
| 0.00 | 3-block, H=48 | 1 | `Full` | `rf_expected_utility` | -0.004975 |
| 0.00 | 3-block, H=48 | 2 | `Medium` | `ridge_expected_utility` | 0.032338 |
| 0.00 | 3-block, H=72 | 0 | `Medium` | `rf_expected_utility` | -0.004843 |
| 0.00 | 3-block, H=72 | 1 | `Small` | `rf_oracle_label` | -0.012107 |
| 0.00 | 3-block, H=72 | 2 | `Tiny` | `rf_expected_utility` | 0.007264 |
| 0.01 | 3-block, H=36 | 0 | `Small` | `rf_oracle_label` | -0.005647 |
| 0.01 | 3-block, H=36 | 1 | `Medium` | `logreg_oracle_label` | 0.005111 |
| 0.01 | 3-block, H=36 | 2 | `Medium` | `rf_expected_utility` | 0.015625 |
| 0.01 | 3-block, H=48 | 0 | `Medium` | `ridge_expected_utility` | 0.043770 |
| 0.01 | 3-block, H=48 | 1 | `Full` | `rf_oracle_label` | 0.002827 |
| 0.01 | 3-block, H=48 | 2 | `Medium` | `ridge_expected_utility` | 0.033813 |
| 0.01 | 3-block, H=72 | 0 | `Medium` | `rf_expected_utility` | -0.001743 |
| 0.01 | 3-block, H=72 | 1 | `Small` | `rf_oracle_label` | -0.013366 |
| 0.01 | 3-block, H=72 | 2 | `Tiny` | `rf_expected_utility` | 0.008450 |
| 0.02 | 3-block, H=36 | 0 | `Small` | `rf_oracle_label` | -0.006404 |
| 0.02 | 3-block, H=36 | 1 | `Medium` | `logreg_oracle_label` | 0.005333 |
| 0.02 | 3-block, H=36 | 2 | `Tiny` | `logreg_oracle_label` | 0.011833 |
| 0.02 | 3-block, H=48 | 0 | `Medium` | `ridge_expected_utility` | 0.045351 |
| 0.02 | 3-block, H=48 | 1 | `Small` | `rf_oracle_label` | 0.009575 |
| 0.02 | 3-block, H=48 | 2 | `Medium` | `ridge_expected_utility` | 0.032919 |
| 0.02 | 3-block, H=72 | 0 | `Medium` | `ridge_expected_utility` | 0.002655 |
| 0.02 | 3-block, H=72 | 1 | `Small` | `rf_oracle_label` | -0.014625 |
| 0.02 | 3-block, H=72 | 2 | `Tiny` | `rf_expected_utility` | 0.007472 |
| 0.05 | 3-block, H=36 | 0 | `Small` | `rf_expected_utility` | -0.007450 |
| 0.05 | 3-block, H=36 | 1 | `Medium` | `ridge_expected_utility` | 0.009017 |
| 0.05 | 3-block, H=36 | 2 | `Tiny` | `logreg_oracle_label` | 0.007577 |
| 0.05 | 3-block, H=48 | 0 | `Medium` | `ridge_expected_utility` | 0.050159 |
| 0.05 | 3-block, H=48 | 1 | `Small` | `rf_oracle_label` | 0.009012 |
| 0.05 | 3-block, H=48 | 2 | `Medium` | `ridge_expected_utility` | 0.037667 |
| 0.05 | 3-block, H=72 | 0 | `Medium` | `rf_expected_utility` | 0.002837 |
| 0.05 | 3-block, H=72 | 1 | `Small` | `rf_expected_utility` | -0.016960 |
| 0.05 | 3-block, H=72 | 2 | `Tiny` | `rf_expected_utility` | 0.005454 |
| 0.10 | 3-block, H=36 | 0 | `Small` | `rf_expected_utility` | -0.003867 |
| 0.10 | 3-block, H=36 | 1 | `Medium` | `ridge_expected_utility` | 0.015878 |
| 0.10 | 3-block, H=36 | 2 | `Tiny` | `rf_expected_utility` | 0.007767 |
| 0.10 | 3-block, H=48 | 0 | `Tiny` | `ridge_expected_utility` | 0.053612 |
| 0.10 | 3-block, H=48 | 1 | `Small` | `ridge_expected_utility` | 0.011912 |
| 0.10 | 3-block, H=48 | 2 | `Medium` | `ridge_expected_utility` | 0.048166 |
| 0.10 | 3-block, H=72 | 0 | `Medium` | `rf_expected_utility` | 0.020418 |
| 0.10 | 3-block, H=72 | 1 | `Small` | `ridge_expected_utility` | -0.014522 |
| 0.10 | 3-block, H=72 | 2 | `Tiny` | `ridge_expected_utility` | -0.010088 |
| 0.15 | 3-block, H=36 | 0 | `Small` | `ridge_expected_utility` | -0.004700 |
| 0.15 | 3-block, H=36 | 1 | `Medium` | `ridge_expected_utility` | 0.021413 |
| 0.15 | 3-block, H=36 | 2 | `Tiny` | `rf_expected_utility` | 0.004769 |
| 0.15 | 3-block, H=48 | 0 | `Tiny` | `ridge_expected_utility` | 0.052324 |
| 0.15 | 3-block, H=48 | 1 | `Small` | `ridge_expected_utility` | 0.015904 |
| 0.15 | 3-block, H=48 | 2 | `Small` | `ridge_expected_utility` | 0.054074 |
| 0.15 | 3-block, H=72 | 0 | `Tiny` | `rf_expected_utility` | 0.026295 |
| 0.15 | 3-block, H=72 | 1 | `Small` | `ridge_expected_utility` | -0.006514 |
| 0.15 | 3-block, H=72 | 2 | `Tiny` | `ridge_expected_utility` | -0.001836 |
| 0.20 | 3-block, H=36 | 0 | `Tiny` | `rf_expected_utility` | -0.002811 |
| 0.20 | 3-block, H=36 | 1 | `Tiny` | `ridge_expected_utility` | 0.024617 |
| 0.20 | 3-block, H=36 | 2 | `Tiny` | `rf_expected_utility` | 0.003557 |
| 0.20 | 3-block, H=48 | 0 | `Tiny` | `ridge_expected_utility` | 0.051363 |
| 0.20 | 3-block, H=48 | 1 | `Small` | `ridge_expected_utility` | 0.015248 |
| 0.20 | 3-block, H=48 | 2 | `Tiny` | `ridge_expected_utility` | 0.053160 |
| 0.20 | 3-block, H=72 | 0 | `Tiny` | `rf_expected_utility` | 0.027417 |
| 0.20 | 3-block, H=72 | 1 | `Small` | `ridge_expected_utility` | -0.003385 |
| 0.20 | 3-block, H=72 | 2 | `Tiny` | `ridge_expected_utility` | -0.003449 |
| 0.30 | 3-block, H=36 | 0 | `Tiny` | `rf_expected_utility` | -0.002367 |
| 0.30 | 3-block, H=36 | 1 | `Tiny` | `rf_expected_utility` | 0.027679 |
| 0.30 | 3-block, H=36 | 2 | `Tiny` | `rf_expected_utility` | -0.001146 |
| 0.30 | 3-block, H=48 | 0 | `Tiny` | `ridge_expected_utility` | 0.043232 |
| 0.30 | 3-block, H=48 | 1 | `Small` | `ridge_expected_utility` | 0.018061 |
| 0.30 | 3-block, H=48 | 2 | `Tiny` | `ridge_expected_utility` | 0.057549 |
| 0.30 | 3-block, H=72 | 0 | `Tiny` | `rf_expected_utility` | 0.018115 |
| 0.30 | 3-block, H=72 | 1 | `Small` | `ridge_expected_utility` | -0.002114 |
| 0.30 | 3-block, H=72 | 2 | `Tiny` | `rf_expected_utility` | -0.008163 |

## Interpretation

- This is the robust version of v27/v28: routing is evaluated over seed-aligned capacity ladders instead of a single seed0 ladder.
- If learned-fixed remains positive on H=48, the routing result survives multi-seed capacity retraining.
- If H=72 keeps a large oracle gap, it remains a routing-signal problem rather than a solved regime.

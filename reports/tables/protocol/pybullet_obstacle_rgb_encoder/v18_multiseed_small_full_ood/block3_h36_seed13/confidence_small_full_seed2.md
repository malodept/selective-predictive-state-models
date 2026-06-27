# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.785306 |
| biased top-1 | 0.884000 |
| strict top-1 | 0.702000 |
| tie-aware top-1 | 0.762667 |
| correct tied with another candidate | 0.182000 |
| mean tie count | 1.364000 |
| ECE vs tie-aware target | 0.032501 |
| ECE vs strict target | 0.093168 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.785306 | 0.884000 | 0.702000 | 0.762667 | 0.182000 | 1.364000 |
| 0.90 | 450 | 0.836256 | 0.873333 | 0.780000 | 0.811111 | 0.093333 | 1.186667 |
| 0.80 | 400 | 0.897848 | 0.875000 | 0.875000 | 0.875000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.946544 | 0.942857 | 0.942857 | 0.942857 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.974767 | 0.980000 | 0.980000 | 0.980000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.988356 | 0.988000 | 0.988000 | 0.988000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.995133 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998462 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 91 | 0.329601 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 99 | 0.829263 | 0.838384 | 0.838384 | 0.838384 | 0.000000 | 1.000000 |
| left | 120 | 0.857106 | 0.766667 | 0.766667 | 0.766667 | 0.000000 | 1.000000 |
| forward | 97 | 0.919109 | 0.886598 | 0.886598 | 0.886598 | 0.000000 | 1.000000 |
| backward | 93 | 0.952214 | 0.967742 | 0.967742 | 0.967742 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 94 | 0.330030 | 0.322695 | 0.007335 |
| [0.4,0.5] | 16 | 0.448443 | 0.187500 | 0.260943 |
| [0.5,0.6] | 24 | 0.543441 | 0.416667 | 0.126775 |
| [0.6,0.7] | 23 | 0.645252 | 0.521739 | 0.123512 |
| [0.7,0.8] | 26 | 0.762716 | 0.730769 | 0.031947 |
| [0.8,0.9] | 35 | 0.863044 | 0.800000 | 0.063044 |
| [0.9,1.0] | 282 | 0.980619 | 0.989362 | 0.008743 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

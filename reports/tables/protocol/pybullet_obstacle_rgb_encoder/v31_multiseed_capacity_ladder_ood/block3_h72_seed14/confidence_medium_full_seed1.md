# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.792252 |
| biased top-1 | 0.862000 |
| strict top-1 | 0.688000 |
| tie-aware top-1 | 0.746000 |
| correct tied with another candidate | 0.174000 |
| mean tie count | 1.348000 |
| ECE vs tie-aware target | 0.049170 |
| ECE vs strict target | 0.107170 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.792252 | 0.862000 | 0.688000 | 0.746000 | 0.174000 | 1.348000 |
| 0.90 | 450 | 0.843904 | 0.848889 | 0.764444 | 0.792593 | 0.084444 | 1.168889 |
| 0.80 | 400 | 0.903546 | 0.847500 | 0.847500 | 0.847500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.946113 | 0.905714 | 0.905714 | 0.905714 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.975819 | 0.960000 | 0.960000 | 0.960000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.990584 | 0.980000 | 0.980000 | 0.980000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.996079 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998561 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 87 | 0.330149 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 107 | 0.890023 | 0.794393 | 0.794393 | 0.794393 | 0.000000 | 1.000000 |
| left | 102 | 0.827046 | 0.774510 | 0.774510 | 0.774510 | 0.000000 | 1.000000 |
| forward | 106 | 0.901096 | 0.839623 | 0.839623 | 0.839623 | 0.000000 | 1.000000 |
| backward | 98 | 0.941795 | 0.928571 | 0.928571 | 0.928571 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.289190 | 0.000000 | 0.289190 |
| [0.3,0.4] | 88 | 0.330882 | 0.329545 | 0.001336 |
| [0.4,0.5] | 7 | 0.467225 | 0.571429 | 0.104204 |
| [0.5,0.6] | 27 | 0.549518 | 0.407407 | 0.142110 |
| [0.6,0.7] | 28 | 0.648987 | 0.428571 | 0.220415 |
| [0.7,0.8] | 37 | 0.748906 | 0.567568 | 0.181339 |
| [0.8,0.9] | 36 | 0.862144 | 0.805556 | 0.056588 |
| [0.9,1.0] | 276 | 0.984397 | 0.967391 | 0.017005 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

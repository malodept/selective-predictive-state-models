# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.858533 |
| biased top-1 | 0.980000 |
| strict top-1 | 0.810000 |
| tie-aware top-1 | 0.867000 |
| correct tied with another candidate | 0.170000 |
| mean tie count | 1.338000 |
| ECE vs tie-aware target | 0.011715 |
| ECE vs strict target | 0.066972 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.858534 | 0.980000 | 0.810000 | 0.867000 | 0.168000 | 1.338000 |
| 0.90 | 450 | 0.917204 | 0.977778 | 0.900000 | 0.926296 | 0.075556 | 1.153333 |
| 0.80 | 400 | 0.979313 | 0.987500 | 0.987500 | 0.987500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.993903 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.997856 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999178 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999673 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999871 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 84 | 0.331613 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 97 | 0.953864 | 0.979381 | 0.979381 | 0.979381 | 0.000000 | 1.000000 |
| left | 100 | 0.937948 | 0.950000 | 0.950000 | 0.950000 | 0.000000 | 1.000000 |
| forward | 114 | 0.986073 | 0.991228 | 0.991228 | 0.991228 | 0.000000 | 1.000000 |
| backward | 105 | 0.977898 | 0.980952 | 0.971429 | 0.976190 | 0.009524 | 1.009524 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 86 | 0.332201 | 0.337209 | 0.005009 |
| [0.4,0.5] | 1 | 0.494984 | 0.500000 | 0.005016 |
| [0.5,0.6] | 3 | 0.552456 | 0.666667 | 0.114211 |
| [0.6,0.7] | 6 | 0.635353 | 0.500000 | 0.135353 |
| [0.7,0.8] | 9 | 0.768462 | 0.888889 | 0.120427 |
| [0.8,0.9] | 25 | 0.855928 | 0.920000 | 0.064072 |
| [0.9,1.0] | 370 | 0.990321 | 0.994595 | 0.004274 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

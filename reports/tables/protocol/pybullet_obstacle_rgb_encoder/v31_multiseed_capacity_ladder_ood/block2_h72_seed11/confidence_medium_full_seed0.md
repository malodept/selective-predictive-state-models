# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.851392 |
| biased top-1 | 0.964000 |
| strict top-1 | 0.794000 |
| tie-aware top-1 | 0.851000 |
| correct tied with another candidate | 0.170000 |
| mean tie count | 1.338000 |
| ECE vs tie-aware target | 0.017515 |
| ECE vs strict target | 0.073386 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.851392 | 0.964000 | 0.794000 | 0.851000 | 0.168000 | 1.338000 |
| 0.90 | 450 | 0.909308 | 0.964444 | 0.882222 | 0.910000 | 0.080000 | 1.162222 |
| 0.80 | 400 | 0.974667 | 0.977500 | 0.977500 | 0.977500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.992842 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.997446 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999153 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999649 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999856 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 84 | 0.331611 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 97 | 0.953983 | 0.948454 | 0.948454 | 0.948454 | 0.000000 | 1.000000 |
| left | 100 | 0.918401 | 0.900000 | 0.900000 | 0.900000 | 0.000000 | 1.000000 |
| forward | 114 | 0.981759 | 0.982456 | 0.982456 | 0.982456 | 0.000000 | 1.000000 |
| backward | 105 | 0.967081 | 0.990476 | 0.980952 | 0.985714 | 0.009524 | 1.009524 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 86 | 0.331442 | 0.325581 | 0.005861 |
| [0.4,0.5] | 5 | 0.443565 | 0.500000 | 0.056435 |
| [0.5,0.6] | 7 | 0.556794 | 0.428571 | 0.128222 |
| [0.6,0.7] | 4 | 0.637979 | 0.500000 | 0.137979 |
| [0.7,0.8] | 10 | 0.752315 | 0.500000 | 0.252315 |
| [0.8,0.9] | 20 | 0.844978 | 0.850000 | 0.005022 |
| [0.9,1.0] | 368 | 0.989407 | 1.000000 | 0.010593 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

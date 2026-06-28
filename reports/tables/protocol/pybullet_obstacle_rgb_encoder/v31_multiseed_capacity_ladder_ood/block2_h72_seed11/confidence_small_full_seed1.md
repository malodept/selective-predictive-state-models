# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.855644 |
| biased top-1 | 0.980000 |
| strict top-1 | 0.810000 |
| tie-aware top-1 | 0.867000 |
| correct tied with another candidate | 0.170000 |
| mean tie count | 1.338000 |
| ECE vs tie-aware target | 0.015602 |
| ECE vs strict target | 0.072284 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.855644 | 0.980000 | 0.810000 | 0.867000 | 0.168000 | 1.338000 |
| 0.90 | 450 | 0.913987 | 0.980000 | 0.900000 | 0.927037 | 0.077778 | 1.157778 |
| 0.80 | 400 | 0.978356 | 0.990000 | 0.990000 | 0.990000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.995003 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.998099 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999194 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999645 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999859 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 84 | 0.331682 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 97 | 0.962326 | 0.979381 | 0.979381 | 0.979381 | 0.000000 | 1.000000 |
| left | 100 | 0.917926 | 0.930000 | 0.930000 | 0.930000 | 0.000000 | 1.000000 |
| forward | 114 | 0.980751 | 0.991228 | 0.991228 | 0.991228 | 0.000000 | 1.000000 |
| backward | 105 | 0.981112 | 1.000000 | 0.990476 | 0.995238 | 0.009524 | 1.009524 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 86 | 0.332157 | 0.325581 | 0.006576 |
| [0.4,0.5] | 3 | 0.473510 | 0.500000 | 0.026490 |
| [0.5,0.6] | 6 | 0.553919 | 0.833333 | 0.279414 |
| [0.6,0.7] | 7 | 0.642281 | 0.571429 | 0.070852 |
| [0.7,0.8] | 9 | 0.757419 | 0.888889 | 0.131470 |
| [0.8,0.9] | 18 | 0.857865 | 1.000000 | 0.142135 |
| [0.9,1.0] | 371 | 0.991261 | 0.994609 | 0.003348 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

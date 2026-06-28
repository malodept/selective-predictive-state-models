# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.863623 |
| biased top-1 | 0.980000 |
| strict top-1 | 0.810000 |
| tie-aware top-1 | 0.867000 |
| correct tied with another candidate | 0.170000 |
| mean tie count | 1.338000 |
| ECE vs tie-aware target | 0.016225 |
| ECE vs strict target | 0.067830 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.863623 | 0.980000 | 0.810000 | 0.867000 | 0.168000 | 1.338000 |
| 0.90 | 450 | 0.922807 | 0.977778 | 0.900000 | 0.926296 | 0.075556 | 1.153333 |
| 0.80 | 400 | 0.987935 | 0.997500 | 0.997500 | 0.997500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.996053 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.998271 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999245 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999628 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999839 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 84 | 0.331883 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 97 | 0.968642 | 0.979381 | 0.979381 | 0.979381 | 0.000000 | 1.000000 |
| left | 100 | 0.937267 | 0.930000 | 0.930000 | 0.930000 | 0.000000 | 1.000000 |
| forward | 114 | 0.988478 | 0.991228 | 0.991228 | 0.991228 | 0.000000 | 1.000000 |
| backward | 105 | 0.986303 | 1.000000 | 0.990476 | 0.995238 | 0.009524 | 1.009524 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 86 | 0.333156 | 0.348837 | 0.015681 |
| [0.4,0.5] | 5 | 0.463011 | 0.100000 | 0.363011 |
| [0.5,0.6] | 4 | 0.540082 | 0.250000 | 0.290082 |
| [0.6,0.7] | 2 | 0.618239 | 0.500000 | 0.118239 |
| [0.7,0.8] | 5 | 0.760574 | 0.800000 | 0.039426 |
| [0.8,0.9] | 5 | 0.858821 | 1.000000 | 0.141179 |
| [0.9,1.0] | 393 | 0.990716 | 0.997455 | 0.006740 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

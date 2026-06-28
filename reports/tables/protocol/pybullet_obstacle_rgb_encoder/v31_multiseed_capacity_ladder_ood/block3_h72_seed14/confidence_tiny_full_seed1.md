# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.766584 |
| biased top-1 | 0.814000 |
| strict top-1 | 0.640000 |
| tie-aware top-1 | 0.698000 |
| correct tied with another candidate | 0.174000 |
| mean tie count | 1.348000 |
| ECE vs tie-aware target | 0.069776 |
| ECE vs strict target | 0.126583 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.766584 | 0.814000 | 0.640000 | 0.698000 | 0.174000 | 1.348000 |
| 0.90 | 450 | 0.815671 | 0.793333 | 0.711111 | 0.738519 | 0.082222 | 1.164444 |
| 0.80 | 400 | 0.872697 | 0.785000 | 0.785000 | 0.785000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.919094 | 0.834286 | 0.834286 | 0.834286 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.956157 | 0.883333 | 0.883333 | 0.883333 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.980734 | 0.924000 | 0.924000 | 0.924000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.992370 | 0.975000 | 0.975000 | 0.975000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.997501 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 87 | 0.328167 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 107 | 0.899767 | 0.878505 | 0.878505 | 0.878505 | 0.000000 | 1.000000 |
| left | 102 | 0.813072 | 0.725490 | 0.725490 | 0.725490 | 0.000000 | 1.000000 |
| forward | 106 | 0.864927 | 0.707547 | 0.707547 | 0.707547 | 0.000000 | 1.000000 |
| backward | 98 | 0.855617 | 0.785714 | 0.785714 | 0.785714 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 3 | 0.274881 | 0.333333 | 0.058452 |
| [0.3,0.4] | 87 | 0.331923 | 0.333333 | 0.001410 |
| [0.4,0.5] | 20 | 0.467067 | 0.400000 | 0.067067 |
| [0.5,0.6] | 35 | 0.557678 | 0.457143 | 0.100535 |
| [0.6,0.7] | 29 | 0.639665 | 0.551724 | 0.087940 |
| [0.7,0.8] | 42 | 0.760266 | 0.619048 | 0.141218 |
| [0.8,0.9] | 32 | 0.851952 | 0.687500 | 0.164452 |
| [0.9,1.0] | 252 | 0.980103 | 0.916667 | 0.063436 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

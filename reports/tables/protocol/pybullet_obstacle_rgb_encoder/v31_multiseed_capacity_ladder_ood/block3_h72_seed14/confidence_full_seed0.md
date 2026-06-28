# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.798075 |
| biased top-1 | 0.850000 |
| strict top-1 | 0.676000 |
| tie-aware top-1 | 0.734000 |
| correct tied with another candidate | 0.174000 |
| mean tie count | 1.348000 |
| ECE vs tie-aware target | 0.064076 |
| ECE vs strict target | 0.122076 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.798075 | 0.850000 | 0.676000 | 0.734000 | 0.174000 | 1.348000 |
| 0.90 | 450 | 0.850206 | 0.833333 | 0.751111 | 0.778519 | 0.082222 | 1.164444 |
| 0.80 | 400 | 0.911506 | 0.835000 | 0.835000 | 0.835000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.953212 | 0.894286 | 0.894286 | 0.894286 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.979744 | 0.953333 | 0.953333 | 0.953333 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.991785 | 0.984000 | 0.984000 | 0.984000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.996429 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998530 | 0.993333 | 0.993333 | 0.993333 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 87 | 0.330516 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 107 | 0.891340 | 0.775701 | 0.775701 | 0.775701 | 0.000000 | 1.000000 |
| left | 102 | 0.860801 | 0.774510 | 0.774510 | 0.774510 | 0.000000 | 1.000000 |
| forward | 106 | 0.894384 | 0.773585 | 0.773585 | 0.773585 | 0.000000 | 1.000000 |
| backward | 98 | 0.941868 | 0.959184 | 0.959184 | 0.959184 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 89 | 0.330948 | 0.325843 | 0.005106 |
| [0.4,0.5] | 11 | 0.452801 | 0.363636 | 0.089164 |
| [0.5,0.6] | 19 | 0.559720 | 0.315789 | 0.243930 |
| [0.6,0.7] | 24 | 0.639688 | 0.500000 | 0.139688 |
| [0.7,0.8] | 33 | 0.749521 | 0.393939 | 0.355582 |
| [0.8,0.9] | 34 | 0.846904 | 0.705882 | 0.141021 |
| [0.9,1.0] | 290 | 0.983057 | 0.962069 | 0.020988 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

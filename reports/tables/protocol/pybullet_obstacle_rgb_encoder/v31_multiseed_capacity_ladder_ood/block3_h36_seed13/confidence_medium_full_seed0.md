# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.778457 |
| biased top-1 | 0.888000 |
| strict top-1 | 0.706000 |
| tie-aware top-1 | 0.766667 |
| correct tied with another candidate | 0.182000 |
| mean tie count | 1.364000 |
| ECE vs tie-aware target | 0.020601 |
| ECE vs strict target | 0.081130 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.778457 | 0.888000 | 0.706000 | 0.766667 | 0.182000 | 1.364000 |
| 0.90 | 450 | 0.828606 | 0.875556 | 0.784444 | 0.814815 | 0.091111 | 1.182222 |
| 0.80 | 400 | 0.888859 | 0.872500 | 0.872500 | 0.872500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.939634 | 0.937143 | 0.937143 | 0.937143 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.971024 | 0.980000 | 0.980000 | 0.980000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.985912 | 0.992000 | 0.992000 | 0.992000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.993023 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.997494 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 91 | 0.329613 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 99 | 0.858967 | 0.858586 | 0.858586 | 0.858586 | 0.000000 | 1.000000 |
| left | 120 | 0.859189 | 0.816667 | 0.816667 | 0.816667 | 0.000000 | 1.000000 |
| forward | 97 | 0.911000 | 0.845361 | 0.845361 | 0.845361 | 0.000000 | 1.000000 |
| backward | 93 | 0.889532 | 0.946237 | 0.946237 | 0.946237 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.298897 | 0.333333 | 0.034437 |
| [0.3,0.4] | 94 | 0.332232 | 0.329787 | 0.002445 |
| [0.4,0.5] | 18 | 0.464078 | 0.500000 | 0.035922 |
| [0.5,0.6] | 32 | 0.542925 | 0.375000 | 0.167925 |
| [0.6,0.7] | 16 | 0.646167 | 0.562500 | 0.083667 |
| [0.7,0.8] | 26 | 0.752405 | 0.730769 | 0.021636 |
| [0.8,0.9] | 39 | 0.861366 | 0.846154 | 0.015212 |
| [0.9,1.0] | 274 | 0.979849 | 0.985401 | 0.005553 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

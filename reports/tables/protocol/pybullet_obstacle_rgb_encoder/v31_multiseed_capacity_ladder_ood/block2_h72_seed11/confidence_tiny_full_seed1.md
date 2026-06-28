# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.856935 |
| biased top-1 | 0.974000 |
| strict top-1 | 0.804000 |
| tie-aware top-1 | 0.861000 |
| correct tied with another candidate | 0.170000 |
| mean tie count | 1.338000 |
| ECE vs tie-aware target | 0.013509 |
| ECE vs strict target | 0.069952 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.856935 | 0.974000 | 0.804000 | 0.861000 | 0.168000 | 1.338000 |
| 0.90 | 450 | 0.915417 | 0.971111 | 0.893333 | 0.919630 | 0.075556 | 1.153333 |
| 0.80 | 400 | 0.979344 | 0.990000 | 0.990000 | 0.990000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.994196 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.997453 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.998952 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999634 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999862 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 84 | 0.331674 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 97 | 0.954394 | 0.958763 | 0.958763 | 0.958763 | 0.000000 | 1.000000 |
| left | 100 | 0.935130 | 0.930000 | 0.930000 | 0.930000 | 0.000000 | 1.000000 |
| forward | 114 | 0.985634 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| backward | 105 | 0.972911 | 0.980952 | 0.971429 | 0.976190 | 0.009524 | 1.009524 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 84 | 0.331674 | 0.333333 | 0.001660 |
| [0.4,0.5] | 6 | 0.448466 | 0.250000 | 0.198466 |
| [0.5,0.6] | 4 | 0.541399 | 0.500000 | 0.041399 |
| [0.6,0.7] | 5 | 0.661574 | 0.600000 | 0.061574 |
| [0.7,0.8] | 9 | 0.744107 | 0.666667 | 0.077441 |
| [0.8,0.9] | 19 | 0.857310 | 0.947368 | 0.090058 |
| [0.9,1.0] | 373 | 0.990501 | 0.997319 | 0.006818 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

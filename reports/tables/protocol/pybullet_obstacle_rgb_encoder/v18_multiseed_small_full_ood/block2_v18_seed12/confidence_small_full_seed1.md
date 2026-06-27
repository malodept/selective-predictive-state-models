# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.839738 |
| biased top-1 | 0.984000 |
| strict top-1 | 0.772000 |
| tie-aware top-1 | 0.842667 |
| correct tied with another candidate | 0.212000 |
| mean tie count | 1.424000 |
| ECE vs tie-aware target | 0.007327 |
| ECE vs strict target | 0.075848 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.839738 | 0.984000 | 0.772000 | 0.842667 | 0.212000 | 1.424000 |
| 0.90 | 450 | 0.896654 | 0.982222 | 0.857778 | 0.899259 | 0.124444 | 1.248889 |
| 0.80 | 400 | 0.967110 | 0.980000 | 0.965000 | 0.970000 | 0.015000 | 1.030000 |
| 0.70 | 350 | 0.997146 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.998984 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999559 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999803 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999918 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 106 | 0.330424 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 113 | 0.978627 | 0.991150 | 0.991150 | 0.991150 | 0.000000 | 1.000000 |
| left | 105 | 0.964759 | 0.971429 | 0.971429 | 0.971429 | 0.000000 | 1.000000 |
| forward | 83 | 0.985028 | 0.987952 | 0.987952 | 0.987952 | 0.000000 | 1.000000 |
| backward | 93 | 0.980667 | 0.967742 | 0.967742 | 0.967742 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 108 | 0.331455 | 0.336420 | 0.004965 |
| [0.4,0.5] | 0 | NA | NA | NA |
| [0.5,0.6] | 6 | 0.539758 | 0.500000 | 0.039758 |
| [0.6,0.7] | 3 | 0.620286 | 0.333333 | 0.286952 |
| [0.7,0.8] | 6 | 0.769030 | 0.833333 | 0.064303 |
| [0.8,0.9] | 2 | 0.871463 | 1.000000 | 0.128537 |
| [0.9,1.0] | 375 | 0.993641 | 0.997333 | 0.003693 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

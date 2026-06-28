# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.807835 |
| biased top-1 | 0.900000 |
| strict top-1 | 0.704000 |
| tie-aware top-1 | 0.769333 |
| correct tied with another candidate | 0.196000 |
| mean tie count | 1.392000 |
| ECE vs tie-aware target | 0.041094 |
| ECE vs strict target | 0.103835 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.807835 | 0.900000 | 0.704000 | 0.769333 | 0.196000 | 1.392000 |
| 0.90 | 450 | 0.861292 | 0.891111 | 0.782222 | 0.818519 | 0.108889 | 1.217778 |
| 0.80 | 400 | 0.927247 | 0.877500 | 0.877500 | 0.877500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.971165 | 0.942857 | 0.942857 | 0.942857 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.987782 | 0.980000 | 0.980000 | 0.980000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.994635 | 0.996000 | 0.996000 | 0.996000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.997666 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999067 | 0.993333 | 0.993333 | 0.993333 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 98 | 0.329708 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 78 | 0.910856 | 0.858974 | 0.858974 | 0.858974 | 0.000000 | 1.000000 |
| left | 102 | 0.921889 | 0.852941 | 0.852941 | 0.852941 | 0.000000 | 1.000000 |
| forward | 117 | 0.931467 | 0.897436 | 0.897436 | 0.897436 | 0.000000 | 1.000000 |
| backward | 105 | 0.928997 | 0.885714 | 0.885714 | 0.885714 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.263370 | 0.333333 | 0.069963 |
| [0.3,0.4] | 99 | 0.330859 | 0.336700 | 0.005842 |
| [0.4,0.5] | 6 | 0.461270 | 0.166667 | 0.294603 |
| [0.5,0.6] | 16 | 0.542826 | 0.250000 | 0.292826 |
| [0.6,0.7] | 14 | 0.654298 | 0.500000 | 0.154298 |
| [0.7,0.8] | 20 | 0.754951 | 0.550000 | 0.204951 |
| [0.8,0.9] | 24 | 0.857558 | 0.666667 | 0.190891 |
| [0.9,1.0] | 320 | 0.983142 | 0.975000 | 0.008142 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

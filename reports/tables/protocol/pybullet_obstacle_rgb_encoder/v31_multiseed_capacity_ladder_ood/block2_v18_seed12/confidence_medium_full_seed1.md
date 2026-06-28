# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.845374 |
| biased top-1 | 0.992000 |
| strict top-1 | 0.780000 |
| tie-aware top-1 | 0.850667 |
| correct tied with another candidate | 0.212000 |
| mean tie count | 1.424000 |
| ECE vs tie-aware target | 0.010301 |
| ECE vs strict target | 0.079739 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.845374 | 0.992000 | 0.780000 | 0.850667 | 0.212000 | 1.424000 |
| 0.90 | 450 | 0.902913 | 0.991111 | 0.866667 | 0.908148 | 0.124444 | 1.248889 |
| 0.80 | 400 | 0.974152 | 0.990000 | 0.975000 | 0.980000 | 0.015000 | 1.030000 |
| 0.70 | 350 | 0.998032 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.999154 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999604 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999812 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999914 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 106 | 0.330434 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 113 | 0.978094 | 0.982301 | 0.982301 | 0.982301 | 0.000000 | 1.000000 |
| left | 105 | 0.978017 | 0.980952 | 0.980952 | 0.980952 | 0.000000 | 1.000000 |
| forward | 83 | 0.992640 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| backward | 93 | 0.989846 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 106 | 0.330434 | 0.333333 | 0.002899 |
| [0.4,0.5] | 0 | NA | NA | NA |
| [0.5,0.6] | 3 | 0.564893 | 1.000000 | 0.435107 |
| [0.6,0.7] | 4 | 0.647770 | 0.750000 | 0.102230 |
| [0.7,0.8] | 3 | 0.750735 | 0.333333 | 0.417401 |
| [0.8,0.9] | 6 | 0.847941 | 1.000000 | 0.152059 |
| [0.9,1.0] | 378 | 0.994803 | 0.997354 | 0.002551 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

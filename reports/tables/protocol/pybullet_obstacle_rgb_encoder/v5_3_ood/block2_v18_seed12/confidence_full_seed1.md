# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.846249 |
| biased top-1 | 0.992000 |
| strict top-1 | 0.780000 |
| tie-aware top-1 | 0.850667 |
| correct tied with another candidate | 0.212000 |
| mean tie count | 1.424000 |
| ECE vs tie-aware target | 0.009059 |
| ECE vs strict target | 0.078507 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.846249 | 0.992000 | 0.780000 | 0.850667 | 0.212000 | 1.424000 |
| 0.90 | 450 | 0.903879 | 0.991111 | 0.866667 | 0.908148 | 0.124444 | 1.248889 |
| 0.80 | 400 | 0.975239 | 0.990000 | 0.975000 | 0.980000 | 0.015000 | 1.030000 |
| 0.70 | 350 | 0.998128 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.999274 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999666 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999853 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999930 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 106 | 0.330460 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 113 | 0.988555 | 0.991150 | 0.991150 | 0.991150 | 0.000000 | 1.000000 |
| left | 105 | 0.974787 | 0.980952 | 0.980952 | 0.980952 | 0.000000 | 1.000000 |
| forward | 83 | 0.988063 | 0.987952 | 0.987952 | 0.987952 | 0.000000 | 1.000000 |
| backward | 93 | 0.989538 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 106 | 0.330460 | 0.333333 | 0.002873 |
| [0.4,0.5] | 1 | 0.401508 | 0.000000 | 0.401508 |
| [0.5,0.6] | 4 | 0.529136 | 0.750000 | 0.220864 |
| [0.6,0.7] | 2 | 0.670772 | 1.000000 | 0.329228 |
| [0.7,0.8] | 1 | 0.758619 | 0.000000 | 0.758619 |
| [0.8,0.9] | 3 | 0.859560 | 1.000000 | 0.140440 |
| [0.9,1.0] | 383 | 0.994513 | 0.997389 | 0.002876 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

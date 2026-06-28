# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.784180 |
| biased top-1 | 0.902000 |
| strict top-1 | 0.720000 |
| tie-aware top-1 | 0.780667 |
| correct tied with another candidate | 0.182000 |
| mean tie count | 1.364000 |
| ECE vs tie-aware target | 0.030413 |
| ECE vs strict target | 0.090926 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.784180 | 0.902000 | 0.720000 | 0.780667 | 0.182000 | 1.364000 |
| 0.90 | 450 | 0.835084 | 0.891111 | 0.800000 | 0.830370 | 0.091111 | 1.182222 |
| 0.80 | 400 | 0.895881 | 0.892500 | 0.892500 | 0.892500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.943791 | 0.937143 | 0.937143 | 0.937143 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.973009 | 0.973333 | 0.973333 | 0.973333 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.986171 | 0.992000 | 0.992000 | 0.992000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.994416 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998160 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 91 | 0.328921 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 99 | 0.874746 | 0.909091 | 0.909091 | 0.909091 | 0.000000 | 1.000000 |
| left | 120 | 0.852574 | 0.816667 | 0.816667 | 0.816667 | 0.000000 | 1.000000 |
| forward | 97 | 0.902197 | 0.876289 | 0.876289 | 0.876289 | 0.000000 | 1.000000 |
| backward | 93 | 0.921898 | 0.935484 | 0.935484 | 0.935484 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.295103 | 0.333333 | 0.038231 |
| [0.3,0.4] | 91 | 0.329965 | 0.329670 | 0.000295 |
| [0.4,0.5] | 17 | 0.456079 | 0.529412 | 0.073333 |
| [0.5,0.6] | 26 | 0.552241 | 0.461538 | 0.090702 |
| [0.6,0.7] | 23 | 0.636669 | 0.739130 | 0.102462 |
| [0.7,0.8] | 21 | 0.747822 | 0.809524 | 0.061701 |
| [0.8,0.9] | 41 | 0.856011 | 0.707317 | 0.148694 |
| [0.9,1.0] | 280 | 0.979330 | 0.985714 | 0.006384 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

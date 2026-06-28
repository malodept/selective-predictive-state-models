# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.842326 |
| biased top-1 | 0.988000 |
| strict top-1 | 0.776000 |
| tie-aware top-1 | 0.846667 |
| correct tied with another candidate | 0.212000 |
| mean tie count | 1.424000 |
| ECE vs tie-aware target | 0.010015 |
| ECE vs strict target | 0.079413 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.842326 | 0.988000 | 0.776000 | 0.846667 | 0.212000 | 1.424000 |
| 0.90 | 450 | 0.899545 | 0.986667 | 0.862222 | 0.903704 | 0.124444 | 1.248889 |
| 0.80 | 400 | 0.970366 | 0.985000 | 0.970000 | 0.975000 | 0.015000 | 1.030000 |
| 0.70 | 350 | 0.996832 | 0.997143 | 0.997143 | 0.997143 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.998929 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999553 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999794 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999907 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 106 | 0.330343 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 113 | 0.986170 | 0.982301 | 0.982301 | 0.982301 | 0.000000 | 1.000000 |
| left | 105 | 0.960689 | 0.971429 | 0.971429 | 0.971429 | 0.000000 | 1.000000 |
| forward | 83 | 0.989178 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| backward | 93 | 0.986401 | 0.989247 | 0.989247 | 0.989247 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 106 | 0.330343 | 0.333333 | 0.002990 |
| [0.4,0.5] | 0 | NA | NA | NA |
| [0.5,0.6] | 7 | 0.540747 | 0.428571 | 0.112176 |
| [0.6,0.7] | 1 | 0.633269 | 0.000000 | 0.633269 |
| [0.7,0.8] | 1 | 0.737019 | 1.000000 | 0.262981 |
| [0.8,0.9] | 9 | 0.842323 | 1.000000 | 0.157677 |
| [0.9,1.0] | 376 | 0.993112 | 0.997340 | 0.004228 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

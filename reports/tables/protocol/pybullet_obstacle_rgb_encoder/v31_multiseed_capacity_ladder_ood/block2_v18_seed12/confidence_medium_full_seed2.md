# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.840938 |
| biased top-1 | 0.986000 |
| strict top-1 | 0.774000 |
| tie-aware top-1 | 0.844667 |
| correct tied with another candidate | 0.212000 |
| mean tie count | 1.424000 |
| ECE vs tie-aware target | 0.008108 |
| ECE vs strict target | 0.077487 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.840938 | 0.986000 | 0.774000 | 0.844667 | 0.212000 | 1.424000 |
| 0.90 | 450 | 0.898016 | 0.984444 | 0.860000 | 0.901481 | 0.124444 | 1.248889 |
| 0.80 | 400 | 0.968643 | 0.982500 | 0.967500 | 0.972500 | 0.015000 | 1.030000 |
| 0.70 | 350 | 0.996042 | 0.997143 | 0.997143 | 0.997143 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.998485 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999322 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999668 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999854 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 106 | 0.330296 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 113 | 0.985039 | 0.982301 | 0.982301 | 0.982301 | 0.000000 | 1.000000 |
| left | 105 | 0.963827 | 0.971429 | 0.971429 | 0.971429 | 0.000000 | 1.000000 |
| forward | 83 | 0.983415 | 0.975904 | 0.975904 | 0.975904 | 0.000000 | 1.000000 |
| backward | 93 | 0.981965 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 106 | 0.330296 | 0.333333 | 0.003037 |
| [0.4,0.5] | 1 | 0.472879 | 0.000000 | 0.472879 |
| [0.5,0.6] | 3 | 0.542671 | 0.666667 | 0.123996 |
| [0.6,0.7] | 4 | 0.655448 | 0.500000 | 0.155448 |
| [0.7,0.8] | 5 | 0.748340 | 0.800000 | 0.051660 |
| [0.8,0.9] | 10 | 0.854781 | 0.900000 | 0.045219 |
| [0.9,1.0] | 371 | 0.993114 | 0.997305 | 0.004191 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

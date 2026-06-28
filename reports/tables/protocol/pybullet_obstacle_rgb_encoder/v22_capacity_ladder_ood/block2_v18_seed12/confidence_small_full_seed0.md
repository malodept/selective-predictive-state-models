# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.837284 |
| biased top-1 | 0.990000 |
| strict top-1 | 0.778000 |
| tie-aware top-1 | 0.848667 |
| correct tied with another candidate | 0.212000 |
| mean tie count | 1.424000 |
| ECE vs tie-aware target | 0.011383 |
| ECE vs strict target | 0.080735 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.837284 | 0.990000 | 0.778000 | 0.848667 | 0.212000 | 1.424000 |
| 0.90 | 450 | 0.893968 | 0.988889 | 0.864444 | 0.905926 | 0.124444 | 1.248889 |
| 0.80 | 400 | 0.964093 | 0.987500 | 0.972500 | 0.977500 | 0.015000 | 1.030000 |
| 0.70 | 350 | 0.995596 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.998754 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999578 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999842 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999941 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 106 | 0.330234 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 113 | 0.958292 | 0.982301 | 0.982301 | 0.982301 | 0.000000 | 1.000000 |
| left | 105 | 0.976088 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| forward | 83 | 0.971196 | 0.975904 | 0.975904 | 0.975904 | 0.000000 | 1.000000 |
| backward | 93 | 0.991954 | 0.989247 | 0.989247 | 0.989247 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.299101 | 0.333333 | 0.034232 |
| [0.3,0.4] | 105 | 0.330530 | 0.333333 | 0.002803 |
| [0.4,0.5] | 2 | 0.432636 | 0.500000 | 0.067364 |
| [0.5,0.6] | 3 | 0.528681 | 0.666667 | 0.137986 |
| [0.6,0.7] | 6 | 0.636546 | 0.666667 | 0.030121 |
| [0.7,0.8] | 9 | 0.763981 | 0.888889 | 0.124908 |
| [0.8,0.9] | 5 | 0.863745 | 1.000000 | 0.136255 |
| [0.9,1.0] | 369 | 0.992336 | 1.000000 | 0.007664 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.840256 |
| biased top-1 | 0.994000 |
| strict top-1 | 0.782000 |
| tie-aware top-1 | 0.852667 |
| correct tied with another candidate | 0.212000 |
| mean tie count | 1.424000 |
| ECE vs tie-aware target | 0.014608 |
| ECE vs strict target | 0.084142 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.840256 | 0.994000 | 0.782000 | 0.852667 | 0.212000 | 1.424000 |
| 0.90 | 450 | 0.897177 | 0.993333 | 0.868889 | 0.910370 | 0.124444 | 1.248889 |
| 0.80 | 400 | 0.967694 | 0.992500 | 0.977500 | 0.982500 | 0.015000 | 1.030000 |
| 0.70 | 350 | 0.996882 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.999039 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999595 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999827 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999927 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 106 | 0.330661 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 113 | 0.981545 | 0.991150 | 0.991150 | 0.991150 | 0.000000 | 1.000000 |
| left | 105 | 0.969913 | 0.990476 | 0.990476 | 0.990476 | 0.000000 | 1.000000 |
| forward | 83 | 0.977868 | 0.987952 | 0.987952 | 0.987952 | 0.000000 | 1.000000 |
| backward | 93 | 0.980207 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 106 | 0.330661 | 0.333333 | 0.002672 |
| [0.4,0.5] | 0 | NA | NA | NA |
| [0.5,0.6] | 9 | 0.560626 | 0.777778 | 0.217152 |
| [0.6,0.7] | 2 | 0.630301 | 1.000000 | 0.369699 |
| [0.7,0.8] | 2 | 0.774663 | 0.500000 | 0.274663 |
| [0.8,0.9] | 11 | 0.849837 | 1.000000 | 0.150163 |
| [0.9,1.0] | 370 | 0.994254 | 1.000000 | 0.005746 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

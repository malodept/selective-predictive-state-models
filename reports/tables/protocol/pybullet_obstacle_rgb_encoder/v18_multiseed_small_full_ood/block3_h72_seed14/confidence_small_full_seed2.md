# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.757303 |
| biased top-1 | 0.830000 |
| strict top-1 | 0.656000 |
| tie-aware top-1 | 0.714000 |
| correct tied with another candidate | 0.174000 |
| mean tie count | 1.348000 |
| ECE vs tie-aware target | 0.045259 |
| ECE vs strict target | 0.101303 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.757303 | 0.830000 | 0.656000 | 0.714000 | 0.174000 | 1.348000 |
| 0.90 | 450 | 0.804995 | 0.811111 | 0.728889 | 0.756296 | 0.082222 | 1.164444 |
| 0.80 | 400 | 0.862046 | 0.802500 | 0.802500 | 0.802500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.909698 | 0.865714 | 0.865714 | 0.865714 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.949849 | 0.926667 | 0.926667 | 0.926667 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.978853 | 0.948000 | 0.948000 | 0.948000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.990745 | 0.980000 | 0.980000 | 0.980000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.996567 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 87 | 0.330096 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 107 | 0.835338 | 0.813084 | 0.813084 | 0.813084 | 0.000000 | 1.000000 |
| left | 102 | 0.784554 | 0.656863 | 0.656863 | 0.656863 | 0.000000 | 1.000000 |
| forward | 106 | 0.868533 | 0.830189 | 0.830189 | 0.830189 | 0.000000 | 1.000000 |
| backward | 98 | 0.902685 | 0.877551 | 0.877551 | 0.877551 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.297379 | 0.333333 | 0.035954 |
| [0.3,0.4] | 91 | 0.332019 | 0.336996 | 0.004977 |
| [0.4,0.5] | 24 | 0.446879 | 0.416667 | 0.030212 |
| [0.5,0.6] | 30 | 0.553583 | 0.333333 | 0.220250 |
| [0.6,0.7] | 41 | 0.645190 | 0.512195 | 0.132995 |
| [0.7,0.8] | 37 | 0.749593 | 0.702703 | 0.046891 |
| [0.8,0.9] | 29 | 0.848036 | 0.827586 | 0.020449 |
| [0.9,1.0] | 247 | 0.979867 | 0.951417 | 0.028450 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

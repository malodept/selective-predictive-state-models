# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.799446 |
| biased top-1 | 0.860000 |
| strict top-1 | 0.678000 |
| tie-aware top-1 | 0.738667 |
| correct tied with another candidate | 0.182000 |
| mean tie count | 1.364000 |
| ECE vs tie-aware target | 0.061126 |
| ECE vs strict target | 0.121657 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.799446 | 0.860000 | 0.678000 | 0.738667 | 0.182000 | 1.364000 |
| 0.90 | 450 | 0.851927 | 0.844444 | 0.753333 | 0.783704 | 0.091111 | 1.182222 |
| 0.80 | 400 | 0.914652 | 0.840000 | 0.840000 | 0.840000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.960460 | 0.911429 | 0.911429 | 0.911429 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.986214 | 0.950000 | 0.950000 | 0.950000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.995145 | 0.988000 | 0.988000 | 0.988000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.998122 | 0.990000 | 0.990000 | 0.990000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999227 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 91 | 0.329554 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 99 | 0.857860 | 0.727273 | 0.727273 | 0.727273 | 0.000000 | 1.000000 |
| left | 120 | 0.895717 | 0.808333 | 0.808333 | 0.808333 | 0.000000 | 1.000000 |
| forward | 97 | 0.917142 | 0.835052 | 0.835052 | 0.835052 | 0.000000 | 1.000000 |
| backward | 93 | 0.950072 | 0.956989 | 0.956989 | 0.956989 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.299455 | 0.333333 | 0.033878 |
| [0.3,0.4] | 91 | 0.330626 | 0.329670 | 0.000955 |
| [0.4,0.5] | 11 | 0.449748 | 0.454545 | 0.004797 |
| [0.5,0.6] | 24 | 0.552065 | 0.333333 | 0.218732 |
| [0.6,0.7] | 26 | 0.656317 | 0.307692 | 0.348625 |
| [0.7,0.8] | 21 | 0.757180 | 0.714286 | 0.042894 |
| [0.8,0.9] | 28 | 0.860502 | 0.714286 | 0.146216 |
| [0.9,1.0] | 298 | 0.986848 | 0.949664 | 0.037184 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

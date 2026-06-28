# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.782790 |
| biased top-1 | 0.914000 |
| strict top-1 | 0.732000 |
| tie-aware top-1 | 0.792667 |
| correct tied with another candidate | 0.182000 |
| mean tie count | 1.364000 |
| ECE vs tie-aware target | 0.020051 |
| ECE vs strict target | 0.080718 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.782790 | 0.914000 | 0.732000 | 0.792667 | 0.182000 | 1.364000 |
| 0.90 | 450 | 0.833510 | 0.908889 | 0.813333 | 0.845185 | 0.095556 | 1.191111 |
| 0.80 | 400 | 0.894957 | 0.905000 | 0.905000 | 0.905000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.942556 | 0.951429 | 0.951429 | 0.951429 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.969479 | 0.983333 | 0.983333 | 0.983333 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.984592 | 0.996000 | 0.996000 | 0.996000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.994040 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998088 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 91 | 0.329362 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 99 | 0.798467 | 0.797980 | 0.797980 | 0.797980 | 0.000000 | 1.000000 |
| left | 120 | 0.884318 | 0.916667 | 0.916667 | 0.916667 | 0.000000 | 1.000000 |
| forward | 97 | 0.907308 | 0.896907 | 0.896907 | 0.896907 | 0.000000 | 1.000000 |
| backward | 93 | 0.948901 | 0.967742 | 0.967742 | 0.967742 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 97 | 0.331498 | 0.323024 | 0.008474 |
| [0.4,0.5] | 14 | 0.456268 | 0.571429 | 0.115160 |
| [0.5,0.6] | 20 | 0.540102 | 0.500000 | 0.040102 |
| [0.6,0.7] | 23 | 0.648234 | 0.652174 | 0.003940 |
| [0.7,0.8] | 24 | 0.751542 | 0.833333 | 0.081791 |
| [0.8,0.9] | 50 | 0.858391 | 0.840000 | 0.018391 |
| [0.9,1.0] | 272 | 0.978617 | 0.992647 | 0.014030 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

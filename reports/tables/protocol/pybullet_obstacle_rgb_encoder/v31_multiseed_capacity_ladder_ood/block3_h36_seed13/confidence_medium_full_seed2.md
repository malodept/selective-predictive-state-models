# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.784324 |
| biased top-1 | 0.904000 |
| strict top-1 | 0.722000 |
| tie-aware top-1 | 0.782667 |
| correct tied with another candidate | 0.182000 |
| mean tie count | 1.364000 |
| ECE vs tie-aware target | 0.027589 |
| ECE vs strict target | 0.087197 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.784324 | 0.904000 | 0.722000 | 0.782667 | 0.182000 | 1.364000 |
| 0.90 | 450 | 0.835136 | 0.893333 | 0.800000 | 0.831111 | 0.093333 | 1.186667 |
| 0.80 | 400 | 0.896450 | 0.897500 | 0.897500 | 0.897500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.943425 | 0.951429 | 0.951429 | 0.951429 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.972594 | 0.983333 | 0.983333 | 0.983333 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.986638 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.994285 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.997824 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 91 | 0.329692 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 99 | 0.855538 | 0.858586 | 0.858586 | 0.858586 | 0.000000 | 1.000000 |
| left | 120 | 0.862713 | 0.883333 | 0.883333 | 0.883333 | 0.000000 | 1.000000 |
| forward | 97 | 0.909108 | 0.855670 | 0.855670 | 0.855670 | 0.000000 | 1.000000 |
| backward | 93 | 0.922072 | 0.935484 | 0.935484 | 0.935484 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.294065 | 0.333333 | 0.039269 |
| [0.3,0.4] | 93 | 0.330908 | 0.333333 | 0.002425 |
| [0.4,0.5] | 17 | 0.452005 | 0.470588 | 0.018584 |
| [0.5,0.6] | 19 | 0.550400 | 0.315789 | 0.234611 |
| [0.6,0.7] | 27 | 0.652620 | 0.703704 | 0.051084 |
| [0.7,0.8] | 26 | 0.746866 | 0.692308 | 0.054558 |
| [0.8,0.9] | 40 | 0.860889 | 0.825000 | 0.035889 |
| [0.9,1.0] | 277 | 0.980061 | 0.996390 | 0.016329 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

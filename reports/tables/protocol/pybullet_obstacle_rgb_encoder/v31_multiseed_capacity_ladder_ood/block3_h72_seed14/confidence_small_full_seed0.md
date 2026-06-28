# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.754733 |
| biased top-1 | 0.778000 |
| strict top-1 | 0.604000 |
| tie-aware top-1 | 0.662000 |
| correct tied with another candidate | 0.174000 |
| mean tie count | 1.348000 |
| ECE vs tie-aware target | 0.095385 |
| ECE vs strict target | 0.150733 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.754733 | 0.778000 | 0.604000 | 0.662000 | 0.174000 | 1.348000 |
| 0.90 | 450 | 0.802251 | 0.755556 | 0.671111 | 0.699259 | 0.084444 | 1.168889 |
| 0.80 | 400 | 0.859196 | 0.742500 | 0.742500 | 0.742500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.906117 | 0.788571 | 0.788571 | 0.788571 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.945114 | 0.840000 | 0.840000 | 0.840000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.972536 | 0.896000 | 0.896000 | 0.896000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.988302 | 0.930000 | 0.930000 | 0.930000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.996216 | 0.966667 | 0.966667 | 0.966667 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 87 | 0.330225 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 107 | 0.820786 | 0.626168 | 0.626168 | 0.626168 | 0.000000 | 1.000000 |
| left | 102 | 0.788432 | 0.745098 | 0.745098 | 0.745098 | 0.000000 | 1.000000 |
| forward | 106 | 0.859606 | 0.669811 | 0.669811 | 0.669811 | 0.000000 | 1.000000 |
| backward | 98 | 0.910963 | 0.897959 | 0.897959 | 0.897959 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.271505 | 0.000000 | 0.271505 |
| [0.3,0.4] | 94 | 0.333371 | 0.340426 | 0.007054 |
| [0.4,0.5] | 18 | 0.457403 | 0.388889 | 0.068514 |
| [0.5,0.6] | 32 | 0.542291 | 0.375000 | 0.167291 |
| [0.6,0.7] | 43 | 0.651221 | 0.465116 | 0.186105 |
| [0.7,0.8] | 35 | 0.757466 | 0.600000 | 0.157466 |
| [0.8,0.9] | 43 | 0.852732 | 0.674419 | 0.178314 |
| [0.9,1.0] | 234 | 0.978591 | 0.897436 | 0.081155 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

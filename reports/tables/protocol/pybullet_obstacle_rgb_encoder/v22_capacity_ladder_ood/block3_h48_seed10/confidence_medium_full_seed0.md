# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.774972 |
| biased top-1 | 0.906000 |
| strict top-1 | 0.710000 |
| tie-aware top-1 | 0.775333 |
| correct tied with another candidate | 0.196000 |
| mean tie count | 1.392000 |
| ECE vs tie-aware target | 0.026527 |
| ECE vs strict target | 0.091533 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.774972 | 0.906000 | 0.710000 | 0.775333 | 0.196000 | 1.392000 |
| 0.90 | 450 | 0.824769 | 0.895556 | 0.788889 | 0.824444 | 0.106667 | 1.213333 |
| 0.80 | 400 | 0.886167 | 0.887500 | 0.887500 | 0.887500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.938163 | 0.934286 | 0.934286 | 0.934286 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.973487 | 0.963333 | 0.963333 | 0.963333 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.987078 | 0.992000 | 0.992000 | 0.992000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.994557 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998072 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 98 | 0.329706 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 78 | 0.927008 | 0.923077 | 0.923077 | 0.923077 | 0.000000 | 1.000000 |
| left | 102 | 0.832182 | 0.803922 | 0.803922 | 0.803922 | 0.000000 | 1.000000 |
| forward | 117 | 0.928069 | 0.931624 | 0.931624 | 0.931624 | 0.000000 | 1.000000 |
| backward | 105 | 0.851443 | 0.876190 | 0.876190 | 0.876190 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.251483 | 0.333333 | 0.081850 |
| [0.3,0.4] | 100 | 0.331401 | 0.323333 | 0.008068 |
| [0.4,0.5] | 13 | 0.450470 | 0.384615 | 0.065855 |
| [0.5,0.6] | 31 | 0.542908 | 0.645161 | 0.102253 |
| [0.6,0.7] | 23 | 0.647486 | 0.521739 | 0.125747 |
| [0.7,0.8] | 26 | 0.751136 | 0.884615 | 0.133479 |
| [0.8,0.9] | 25 | 0.858217 | 0.800000 | 0.058217 |
| [0.9,1.0] | 281 | 0.980537 | 0.978648 | 0.001890 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

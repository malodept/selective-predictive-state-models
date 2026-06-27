# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.785043 |
| biased top-1 | 0.906000 |
| strict top-1 | 0.712000 |
| tie-aware top-1 | 0.776667 |
| correct tied with another candidate | 0.194000 |
| mean tie count | 1.388000 |
| ECE vs tie-aware target | 0.027072 |
| ECE vs strict target | 0.085548 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.785043 | 0.906000 | 0.712000 | 0.776667 | 0.196000 | 1.388000 |
| 0.90 | 450 | 0.836289 | 0.900000 | 0.788889 | 0.825926 | 0.111111 | 1.222222 |
| 0.80 | 400 | 0.899250 | 0.887500 | 0.887500 | 0.887500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.949718 | 0.937143 | 0.937143 | 0.937143 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.978508 | 0.986667 | 0.986667 | 0.986667 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.990060 | 0.996000 | 0.996000 | 0.996000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.995967 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998450 | 0.993333 | 0.993333 | 0.993333 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 98 | 0.329179 | 0.989796 | 0.000000 | 0.329932 | 0.989796 | 2.979592 |
| right | 78 | 0.934001 | 0.923077 | 0.923077 | 0.923077 | 0.000000 | 1.000000 |
| left | 102 | 0.845690 | 0.833333 | 0.833333 | 0.833333 | 0.000000 | 1.000000 |
| forward | 117 | 0.920682 | 0.948718 | 0.948718 | 0.948718 | 0.000000 | 1.000000 |
| backward | 105 | 0.889808 | 0.838095 | 0.838095 | 0.838095 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 3 | 0.271765 | 0.333333 | 0.061568 |
| [0.3,0.4] | 99 | 0.331168 | 0.346801 | 0.015634 |
| [0.4,0.5] | 10 | 0.450892 | 0.300000 | 0.150892 |
| [0.5,0.6] | 23 | 0.548337 | 0.565217 | 0.016880 |
| [0.6,0.7] | 24 | 0.647207 | 0.541667 | 0.105541 |
| [0.7,0.8] | 20 | 0.748364 | 0.650000 | 0.098364 |
| [0.8,0.9] | 29 | 0.857000 | 0.758621 | 0.098379 |
| [0.9,1.0] | 292 | 0.980982 | 0.989726 | 0.008744 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

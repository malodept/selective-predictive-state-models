# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.781496 |
| biased top-1 | 0.892000 |
| strict top-1 | 0.698000 |
| tie-aware top-1 | 0.762667 |
| correct tied with another candidate | 0.194000 |
| mean tie count | 1.388000 |
| ECE vs tie-aware target | 0.019068 |
| ECE vs strict target | 0.083735 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.781496 | 0.892000 | 0.698000 | 0.762667 | 0.196000 | 1.388000 |
| 0.90 | 450 | 0.832118 | 0.882222 | 0.775556 | 0.811111 | 0.106667 | 1.213333 |
| 0.80 | 400 | 0.894223 | 0.870000 | 0.870000 | 0.870000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.941744 | 0.925714 | 0.925714 | 0.925714 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.971029 | 0.966667 | 0.966667 | 0.966667 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.985840 | 0.984000 | 0.984000 | 0.984000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.993345 | 0.990000 | 0.990000 | 0.990000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.997607 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 98 | 0.329155 | 0.989796 | 0.000000 | 0.329932 | 0.989796 | 2.979592 |
| right | 78 | 0.878638 | 0.846154 | 0.846154 | 0.846154 | 0.000000 | 1.000000 |
| left | 102 | 0.878866 | 0.892157 | 0.892157 | 0.892157 | 0.000000 | 1.000000 |
| forward | 117 | 0.912795 | 0.905983 | 0.905983 | 0.905983 | 0.000000 | 1.000000 |
| backward | 105 | 0.890625 | 0.819048 | 0.819048 | 0.819048 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 2 | 0.279276 | 0.166667 | 0.112609 |
| [0.3,0.4] | 97 | 0.330906 | 0.329897 | 0.001009 |
| [0.4,0.5] | 17 | 0.467072 | 0.470588 | 0.003516 |
| [0.5,0.6] | 14 | 0.555517 | 0.500000 | 0.055517 |
| [0.6,0.7] | 27 | 0.649146 | 0.481481 | 0.167664 |
| [0.7,0.8] | 26 | 0.751943 | 0.653846 | 0.098097 |
| [0.8,0.9] | 45 | 0.856203 | 0.844444 | 0.011759 |
| [0.9,1.0] | 272 | 0.980762 | 0.977941 | 0.002821 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.789640 |
| biased top-1 | 0.880000 |
| strict top-1 | 0.686000 |
| tie-aware top-1 | 0.750667 |
| correct tied with another candidate | 0.194000 |
| mean tie count | 1.388000 |
| ECE vs tie-aware target | 0.043940 |
| ECE vs strict target | 0.108606 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.789640 | 0.880000 | 0.686000 | 0.750667 | 0.196000 | 1.388000 |
| 0.90 | 450 | 0.841087 | 0.868889 | 0.762222 | 0.797778 | 0.106667 | 1.213333 |
| 0.80 | 400 | 0.904540 | 0.857500 | 0.857500 | 0.857500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.950834 | 0.914286 | 0.914286 | 0.914286 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.973650 | 0.960000 | 0.960000 | 0.960000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.986016 | 0.976000 | 0.976000 | 0.976000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.993638 | 0.985000 | 0.985000 | 0.985000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.997760 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 98 | 0.329614 | 0.989796 | 0.000000 | 0.329932 | 0.989796 | 2.979592 |
| right | 78 | 0.886815 | 0.833333 | 0.833333 | 0.833333 | 0.000000 | 1.000000 |
| left | 102 | 0.882034 | 0.833333 | 0.833333 | 0.833333 | 0.000000 | 1.000000 |
| forward | 117 | 0.899361 | 0.863248 | 0.863248 | 0.863248 | 0.000000 | 1.000000 |
| backward | 105 | 0.934796 | 0.876190 | 0.876190 | 0.876190 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 2 | 0.277560 | 0.166667 | 0.110894 |
| [0.3,0.4] | 98 | 0.331112 | 0.326531 | 0.004581 |
| [0.4,0.5] | 8 | 0.469811 | 0.625000 | 0.155189 |
| [0.5,0.6] | 21 | 0.542452 | 0.238095 | 0.304356 |
| [0.6,0.7] | 20 | 0.657990 | 0.600000 | 0.057990 |
| [0.7,0.8] | 19 | 0.765512 | 0.684211 | 0.081301 |
| [0.8,0.9] | 44 | 0.852247 | 0.636364 | 0.215883 |
| [0.9,1.0] | 288 | 0.977301 | 0.972222 | 0.005079 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

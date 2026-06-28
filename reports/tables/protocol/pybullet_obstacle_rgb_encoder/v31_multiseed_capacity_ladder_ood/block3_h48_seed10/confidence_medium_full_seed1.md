# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.809631 |
| biased top-1 | 0.906000 |
| strict top-1 | 0.712000 |
| tie-aware top-1 | 0.776667 |
| correct tied with another candidate | 0.194000 |
| mean tie count | 1.388000 |
| ECE vs tie-aware target | 0.034846 |
| ECE vs strict target | 0.098444 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.809631 | 0.906000 | 0.712000 | 0.776667 | 0.196000 | 1.388000 |
| 0.90 | 450 | 0.863294 | 0.897778 | 0.791111 | 0.826667 | 0.106667 | 1.213333 |
| 0.80 | 400 | 0.929102 | 0.885000 | 0.885000 | 0.885000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.969881 | 0.951429 | 0.951429 | 0.951429 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.985603 | 0.980000 | 0.980000 | 0.980000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.993047 | 0.988000 | 0.988000 | 0.988000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.996888 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998893 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 98 | 0.329610 | 0.989796 | 0.000000 | 0.329932 | 0.989796 | 2.979592 |
| right | 78 | 0.929295 | 0.923077 | 0.923077 | 0.923077 | 0.000000 | 1.000000 |
| left | 102 | 0.900140 | 0.843137 | 0.843137 | 0.843137 | 0.000000 | 1.000000 |
| forward | 117 | 0.933166 | 0.914530 | 0.914530 | 0.914530 | 0.000000 | 1.000000 |
| backward | 105 | 0.943183 | 0.866667 | 0.866667 | 0.866667 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 2 | 0.284529 | 0.166667 | 0.117862 |
| [0.3,0.4] | 96 | 0.330549 | 0.333333 | 0.002785 |
| [0.4,0.5] | 8 | 0.474613 | 0.500000 | 0.025387 |
| [0.5,0.6] | 13 | 0.555245 | 0.461538 | 0.093706 |
| [0.6,0.7] | 11 | 0.652353 | 0.454545 | 0.197807 |
| [0.7,0.8] | 19 | 0.740206 | 0.421053 | 0.319153 |
| [0.8,0.9] | 34 | 0.852680 | 0.676471 | 0.176209 |
| [0.9,1.0] | 317 | 0.981917 | 0.977918 | 0.003999 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

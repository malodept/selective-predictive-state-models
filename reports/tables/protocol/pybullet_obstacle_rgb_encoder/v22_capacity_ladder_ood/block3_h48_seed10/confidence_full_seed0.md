# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.799313 |
| biased top-1 | 0.870000 |
| strict top-1 | 0.676000 |
| tie-aware top-1 | 0.740667 |
| correct tied with another candidate | 0.194000 |
| mean tie count | 1.388000 |
| ECE vs tie-aware target | 0.066466 |
| ECE vs strict target | 0.127211 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.799313 | 0.870000 | 0.676000 | 0.740667 | 0.196000 | 1.388000 |
| 0.90 | 450 | 0.851831 | 0.857778 | 0.748889 | 0.785185 | 0.108889 | 1.217778 |
| 0.80 | 400 | 0.916492 | 0.842500 | 0.842500 | 0.842500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.961355 | 0.908571 | 0.908571 | 0.908571 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.984780 | 0.973333 | 0.973333 | 0.973333 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.994164 | 0.996000 | 0.996000 | 0.996000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.997627 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999223 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 98 | 0.329785 | 0.989796 | 0.000000 | 0.329932 | 0.989796 | 2.979592 |
| right | 78 | 0.927567 | 0.884615 | 0.884615 | 0.884615 | 0.000000 | 1.000000 |
| left | 102 | 0.879199 | 0.735294 | 0.735294 | 0.735294 | 0.000000 | 1.000000 |
| forward | 117 | 0.914808 | 0.871795 | 0.871795 | 0.871795 | 0.000000 | 1.000000 |
| backward | 105 | 0.935966 | 0.876190 | 0.876190 | 0.876190 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.281090 | 0.000000 | 0.281090 |
| [0.3,0.4] | 98 | 0.330131 | 0.340136 | 0.010005 |
| [0.4,0.5] | 8 | 0.467108 | 0.125000 | 0.342108 |
| [0.5,0.6] | 18 | 0.556977 | 0.611111 | 0.054134 |
| [0.6,0.7] | 19 | 0.655765 | 0.263158 | 0.392607 |
| [0.7,0.8] | 23 | 0.755252 | 0.260870 | 0.494382 |
| [0.8,0.9] | 33 | 0.848355 | 0.666667 | 0.181688 |
| [0.9,1.0] | 300 | 0.984780 | 0.973333 | 0.011447 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

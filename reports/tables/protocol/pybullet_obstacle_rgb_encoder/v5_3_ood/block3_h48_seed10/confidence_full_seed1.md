# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.800859 |
| biased top-1 | 0.918000 |
| strict top-1 | 0.724000 |
| tie-aware top-1 | 0.788667 |
| correct tied with another candidate | 0.194000 |
| mean tie count | 1.388000 |
| ECE vs tie-aware target | 0.018492 |
| ECE vs strict target | 0.083159 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.800859 | 0.918000 | 0.724000 | 0.788667 | 0.196000 | 1.388000 |
| 0.90 | 450 | 0.853531 | 0.911111 | 0.804444 | 0.840000 | 0.106667 | 1.213333 |
| 0.80 | 400 | 0.918383 | 0.902500 | 0.902500 | 0.902500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.962935 | 0.945714 | 0.945714 | 0.945714 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.982537 | 0.973333 | 0.973333 | 0.973333 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.991569 | 0.984000 | 0.984000 | 0.984000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.996208 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998713 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 98 | 0.329702 | 0.989796 | 0.000000 | 0.329932 | 0.989796 | 2.979592 |
| right | 78 | 0.958441 | 0.923077 | 0.923077 | 0.923077 | 0.000000 | 1.000000 |
| left | 102 | 0.922720 | 0.921569 | 0.921569 | 0.921569 | 0.000000 | 1.000000 |
| forward | 117 | 0.911355 | 0.923077 | 0.923077 | 0.923077 | 0.000000 | 1.000000 |
| backward | 105 | 0.882040 | 0.838095 | 0.838095 | 0.838095 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.261438 | 0.000000 | 0.261438 |
| [0.3,0.4] | 98 | 0.330730 | 0.329932 | 0.000798 |
| [0.4,0.5] | 11 | 0.452907 | 0.454545 | 0.001638 |
| [0.5,0.6] | 15 | 0.557709 | 0.466667 | 0.091042 |
| [0.6,0.7] | 16 | 0.670883 | 0.750000 | 0.079117 |
| [0.7,0.8] | 22 | 0.762267 | 0.636364 | 0.125904 |
| [0.8,0.9] | 31 | 0.861578 | 0.870968 | 0.009390 |
| [0.9,1.0] | 306 | 0.981032 | 0.970588 | 0.010444 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

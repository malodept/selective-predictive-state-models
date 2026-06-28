# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.842369 |
| biased top-1 | 0.984000 |
| strict top-1 | 0.772000 |
| tie-aware top-1 | 0.842667 |
| correct tied with another candidate | 0.212000 |
| mean tie count | 1.424000 |
| ECE vs tie-aware target | 0.014505 |
| ECE vs strict target | 0.083905 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.842369 | 0.984000 | 0.772000 | 0.842667 | 0.212000 | 1.424000 |
| 0.90 | 450 | 0.899602 | 0.982222 | 0.857778 | 0.899259 | 0.124444 | 1.248889 |
| 0.80 | 400 | 0.970419 | 0.980000 | 0.965000 | 0.970000 | 0.015000 | 1.030000 |
| 0.70 | 350 | 0.996658 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.998748 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999430 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999735 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999898 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 106 | 0.330347 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 113 | 0.978611 | 0.973451 | 0.973451 | 0.973451 | 0.000000 | 1.000000 |
| left | 105 | 0.978293 | 0.971429 | 0.971429 | 0.971429 | 0.000000 | 1.000000 |
| forward | 83 | 0.982284 | 0.975904 | 0.975904 | 0.975904 | 0.000000 | 1.000000 |
| backward | 93 | 0.982086 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.298932 | 0.333333 | 0.034401 |
| [0.3,0.4] | 105 | 0.330646 | 0.333333 | 0.002687 |
| [0.4,0.5] | 1 | 0.487308 | 0.000000 | 0.487308 |
| [0.5,0.6] | 4 | 0.553541 | 0.500000 | 0.053541 |
| [0.6,0.7] | 2 | 0.662634 | 1.000000 | 0.337366 |
| [0.7,0.8] | 4 | 0.744763 | 1.000000 | 0.255237 |
| [0.8,0.9] | 8 | 0.856277 | 0.500000 | 0.356277 |
| [0.9,1.0] | 375 | 0.992831 | 0.997333 | 0.004503 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

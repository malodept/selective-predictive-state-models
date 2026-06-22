# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.837703 |
| biased top-1 | 0.942000 |
| strict top-1 | 0.772000 |
| tie-aware top-1 | 0.829000 |
| correct tied with another candidate | 0.170000 |
| mean tie count | 1.338000 |
| ECE vs tie-aware target | 0.020070 |
| ECE vs strict target | 0.073407 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.837703 | 0.942000 | 0.772000 | 0.829000 | 0.168000 | 1.338000 |
| 0.90 | 450 | 0.894035 | 0.935556 | 0.857778 | 0.884074 | 0.075556 | 1.153333 |
| 0.80 | 400 | 0.957734 | 0.937500 | 0.937500 | 0.937500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.986251 | 0.982857 | 0.982857 | 0.982857 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.995824 | 0.993333 | 0.993333 | 0.993333 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.998597 | 0.992000 | 0.992000 | 0.992000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999504 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999856 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 84 | 0.331738 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 97 | 0.920413 | 0.876289 | 0.876289 | 0.876289 | 0.000000 | 1.000000 |
| left | 100 | 0.915478 | 0.910000 | 0.910000 | 0.910000 | 0.000000 | 1.000000 |
| forward | 114 | 0.944448 | 0.929825 | 0.929825 | 0.929825 | 0.000000 | 1.000000 |
| backward | 105 | 0.976103 | 1.000000 | 0.990476 | 0.995238 | 0.009524 | 1.009524 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 86 | 0.332374 | 0.337209 | 0.004835 |
| [0.4,0.5] | 5 | 0.481387 | 0.900000 | 0.418613 |
| [0.5,0.6] | 14 | 0.547660 | 0.571429 | 0.023769 |
| [0.6,0.7] | 9 | 0.651341 | 0.444444 | 0.206897 |
| [0.7,0.8] | 15 | 0.759144 | 0.600000 | 0.159144 |
| [0.8,0.9] | 32 | 0.861552 | 0.812500 | 0.049052 |
| [0.9,1.0] | 339 | 0.989305 | 0.985251 | 0.004055 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

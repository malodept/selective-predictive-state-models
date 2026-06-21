# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `1000`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.872830 |
| biased top-1 | 0.997000 |
| strict top-1 | 0.816000 |
| tie-aware top-1 | 0.876333 |
| correct tied with another candidate | 0.181000 |
| mean tie count | 1.362000 |
| ECE vs tie-aware target | 0.004109 |
| ECE vs strict target | 0.063629 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 1000 | 0.872830 | 0.997000 | 0.816000 | 0.876333 | 0.181000 | 1.362000 |
| 0.90 | 900 | 0.933194 | 0.996667 | 0.906667 | 0.936667 | 0.090000 | 1.180000 |
| 0.80 | 800 | 0.997161 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.70 | 700 | 0.999239 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 600 | 0.999642 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 500 | 0.999806 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 400 | 0.999886 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 300 | 0.999933 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 181 | 0.331087 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 192 | 0.988285 | 0.984375 | 0.984375 | 0.984375 | 0.000000 | 1.000000 |
| left | 205 | 0.991550 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| forward | 200 | 0.994269 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| backward | 222 | 0.995633 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 181 | 0.331087 | 0.333333 | 0.002246 |
| [0.4,0.5] | 2 | 0.447616 | 0.500000 | 0.052384 |
| [0.5,0.6] | 0 | NA | NA | NA |
| [0.6,0.7] | 2 | 0.651233 | 0.500000 | 0.151233 |
| [0.7,0.8] | 1 | 0.784360 | 1.000000 | 0.215640 |
| [0.8,0.9] | 10 | 0.852522 | 0.900000 | 0.047478 |
| [0.9,1.0] | 804 | 0.996761 | 1.000000 | 0.003239 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

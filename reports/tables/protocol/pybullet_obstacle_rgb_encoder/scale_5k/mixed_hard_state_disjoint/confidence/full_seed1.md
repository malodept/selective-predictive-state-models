# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `1000`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.872729 |
| biased top-1 | 0.998000 |
| strict top-1 | 0.817000 |
| tie-aware top-1 | 0.877333 |
| correct tied with another candidate | 0.181000 |
| mean tie count | 1.362000 |
| ECE vs tie-aware target | 0.006092 |
| ECE vs strict target | 0.065681 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 1000 | 0.872729 | 0.998000 | 0.817000 | 0.877333 | 0.181000 | 1.362000 |
| 0.90 | 900 | 0.933048 | 0.997778 | 0.907778 | 0.937778 | 0.090000 | 1.180000 |
| 0.80 | 800 | 0.996634 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.70 | 700 | 0.999085 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 600 | 0.999577 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 500 | 0.999761 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 400 | 0.999857 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 300 | 0.999915 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 181 | 0.331277 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 192 | 0.990690 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| left | 205 | 0.989306 | 0.995122 | 0.995122 | 0.995122 | 0.000000 | 1.000000 |
| forward | 200 | 0.997343 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| backward | 222 | 0.992247 | 0.995495 | 0.995495 | 0.995495 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 181 | 0.331277 | 0.333333 | 0.002056 |
| [0.4,0.5] | 1 | 0.494024 | 0.000000 | 0.494024 |
| [0.5,0.6] | 0 | NA | NA | NA |
| [0.6,0.7] | 2 | 0.624797 | 0.500000 | 0.124797 |
| [0.7,0.8] | 4 | 0.726013 | 1.000000 | 0.273987 |
| [0.8,0.9] | 6 | 0.883666 | 1.000000 | 0.116334 |
| [0.9,1.0] | 806 | 0.996052 | 1.000000 | 0.003948 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `1000`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.873579 |
| biased top-1 | 0.999000 |
| strict top-1 | 0.818000 |
| tie-aware top-1 | 0.878333 |
| correct tied with another candidate | 0.181000 |
| mean tie count | 1.362000 |
| ECE vs tie-aware target | 0.005932 |
| ECE vs strict target | 0.065577 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 1000 | 0.873579 | 0.999000 | 0.818000 | 0.878333 | 0.181000 | 1.362000 |
| 0.90 | 900 | 0.933964 | 0.998889 | 0.908889 | 0.938889 | 0.090000 | 1.180000 |
| 0.80 | 800 | 0.997002 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.70 | 700 | 0.999191 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 600 | 0.999643 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 500 | 0.999805 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 400 | 0.999883 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 300 | 0.999930 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 181 | 0.331431 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 192 | 0.991337 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| left | 205 | 0.990264 | 0.995122 | 0.995122 | 0.995122 | 0.000000 | 1.000000 |
| forward | 200 | 0.996218 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| backward | 222 | 0.995523 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 181 | 0.331431 | 0.333333 | 0.001902 |
| [0.4,0.5] | 0 | NA | NA | NA |
| [0.5,0.6] | 1 | 0.589136 | 0.000000 | 0.589136 |
| [0.6,0.7] | 1 | 0.660753 | 1.000000 | 0.339247 |
| [0.7,0.8] | 2 | 0.755032 | 1.000000 | 0.244968 |
| [0.8,0.9] | 10 | 0.861183 | 1.000000 | 0.138817 |
| [0.9,1.0] | 805 | 0.996545 | 1.000000 | 0.003455 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

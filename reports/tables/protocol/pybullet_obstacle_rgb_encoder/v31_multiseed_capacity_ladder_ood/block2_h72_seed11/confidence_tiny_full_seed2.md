# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.851812 |
| biased top-1 | 0.962000 |
| strict top-1 | 0.792000 |
| tie-aware top-1 | 0.849000 |
| correct tied with another candidate | 0.170000 |
| mean tie count | 1.338000 |
| ECE vs tie-aware target | 0.019927 |
| ECE vs strict target | 0.074297 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.851812 | 0.962000 | 0.792000 | 0.849000 | 0.168000 | 1.338000 |
| 0.90 | 450 | 0.909764 | 0.957778 | 0.880000 | 0.906296 | 0.075556 | 1.153333 |
| 0.80 | 400 | 0.974385 | 0.967500 | 0.967500 | 0.967500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.992767 | 0.997143 | 0.997143 | 0.997143 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.997142 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.998776 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999526 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999823 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 84 | 0.331457 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 97 | 0.958502 | 0.938144 | 0.938144 | 0.938144 | 0.000000 | 1.000000 |
| left | 100 | 0.909117 | 0.900000 | 0.900000 | 0.900000 | 0.000000 | 1.000000 |
| forward | 114 | 0.983238 | 0.991228 | 0.991228 | 0.991228 | 0.000000 | 1.000000 |
| backward | 105 | 0.972265 | 0.980952 | 0.971429 | 0.976190 | 0.009524 | 1.009524 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 84 | 0.331457 | 0.333333 | 0.001876 |
| [0.4,0.5] | 7 | 0.456499 | 0.642857 | 0.186358 |
| [0.5,0.6] | 7 | 0.550986 | 0.428571 | 0.122414 |
| [0.6,0.7] | 5 | 0.645816 | 0.400000 | 0.245816 |
| [0.7,0.8] | 10 | 0.745771 | 0.500000 | 0.245771 |
| [0.8,0.9] | 20 | 0.857050 | 0.800000 | 0.057050 |
| [0.9,1.0] | 367 | 0.989600 | 0.997275 | 0.007675 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

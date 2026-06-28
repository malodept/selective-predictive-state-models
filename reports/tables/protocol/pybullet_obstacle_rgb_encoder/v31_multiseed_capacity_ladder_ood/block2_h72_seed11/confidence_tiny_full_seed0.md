# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.851121 |
| biased top-1 | 0.962000 |
| strict top-1 | 0.792000 |
| tie-aware top-1 | 0.849000 |
| correct tied with another candidate | 0.170000 |
| mean tie count | 1.338000 |
| ECE vs tie-aware target | 0.013835 |
| ECE vs strict target | 0.070291 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.851121 | 0.962000 | 0.792000 | 0.849000 | 0.168000 | 1.338000 |
| 0.90 | 450 | 0.908947 | 0.957778 | 0.880000 | 0.906296 | 0.075556 | 1.153333 |
| 0.80 | 400 | 0.972681 | 0.982500 | 0.982500 | 0.982500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.992947 | 0.997143 | 0.997143 | 0.997143 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.997231 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.998735 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999429 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999774 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 84 | 0.331714 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 97 | 0.940667 | 0.896907 | 0.896907 | 0.896907 | 0.000000 | 1.000000 |
| left | 100 | 0.924707 | 0.940000 | 0.940000 | 0.940000 | 0.000000 | 1.000000 |
| forward | 114 | 0.975109 | 0.991228 | 0.991228 | 0.991228 | 0.000000 | 1.000000 |
| backward | 105 | 0.979226 | 0.980952 | 0.971429 | 0.976190 | 0.009524 | 1.009524 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 84 | 0.331714 | 0.333333 | 0.001619 |
| [0.4,0.5] | 5 | 0.474477 | 0.100000 | 0.374477 |
| [0.5,0.6] | 8 | 0.552943 | 0.375000 | 0.177943 |
| [0.6,0.7] | 12 | 0.657779 | 0.666667 | 0.008887 |
| [0.7,0.8] | 7 | 0.744518 | 0.714286 | 0.030232 |
| [0.8,0.9] | 17 | 0.851851 | 0.823529 | 0.028322 |
| [0.9,1.0] | 367 | 0.989956 | 0.997275 | 0.007319 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

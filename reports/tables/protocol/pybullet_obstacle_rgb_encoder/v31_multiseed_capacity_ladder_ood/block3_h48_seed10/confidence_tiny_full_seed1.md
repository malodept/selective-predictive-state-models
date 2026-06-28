# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.768818 |
| biased top-1 | 0.860000 |
| strict top-1 | 0.668000 |
| tie-aware top-1 | 0.732000 |
| correct tied with another candidate | 0.192000 |
| mean tie count | 1.384000 |
| ECE vs tie-aware target | 0.037093 |
| ECE vs strict target | 0.101093 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.768818 | 0.860000 | 0.668000 | 0.732000 | 0.196000 | 1.384000 |
| 0.90 | 450 | 0.818077 | 0.851111 | 0.742222 | 0.778519 | 0.108889 | 1.217778 |
| 0.80 | 400 | 0.878661 | 0.835000 | 0.835000 | 0.835000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.928691 | 0.902857 | 0.902857 | 0.902857 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.964086 | 0.943333 | 0.943333 | 0.943333 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.983165 | 0.984000 | 0.984000 | 0.984000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.992704 | 0.990000 | 0.990000 | 0.990000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.997249 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 98 | 0.329192 | 0.979592 | 0.000000 | 0.326531 | 0.979592 | 2.959184 |
| right | 78 | 0.929058 | 0.948718 | 0.948718 | 0.948718 | 0.000000 | 1.000000 |
| left | 102 | 0.868695 | 0.803922 | 0.803922 | 0.803922 | 0.000000 | 1.000000 |
| forward | 117 | 0.887213 | 0.897436 | 0.897436 | 0.897436 | 0.000000 | 1.000000 |
| backward | 105 | 0.831149 | 0.695238 | 0.695238 | 0.695238 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.275888 | 0.000000 | 0.275888 |
| [0.3,0.4] | 99 | 0.329985 | 0.323232 | 0.006753 |
| [0.4,0.5] | 16 | 0.447170 | 0.250000 | 0.197170 |
| [0.5,0.6] | 25 | 0.547243 | 0.480000 | 0.067243 |
| [0.6,0.7] | 34 | 0.659871 | 0.470588 | 0.189283 |
| [0.7,0.8] | 20 | 0.746564 | 0.750000 | 0.003436 |
| [0.8,0.9] | 46 | 0.853244 | 0.739130 | 0.114114 |
| [0.9,1.0] | 259 | 0.980743 | 0.976834 | 0.003909 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

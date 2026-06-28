# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.783481 |
| biased top-1 | 0.886000 |
| strict top-1 | 0.692000 |
| tie-aware top-1 | 0.756667 |
| correct tied with another candidate | 0.194000 |
| mean tie count | 1.388000 |
| ECE vs tie-aware target | 0.036955 |
| ECE vs strict target | 0.095369 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.783481 | 0.886000 | 0.692000 | 0.756667 | 0.196000 | 1.388000 |
| 0.90 | 450 | 0.834238 | 0.873333 | 0.766667 | 0.802222 | 0.108889 | 1.213333 |
| 0.80 | 400 | 0.896805 | 0.860000 | 0.860000 | 0.860000 | 0.002500 | 1.000000 |
| 0.70 | 350 | 0.945659 | 0.920000 | 0.920000 | 0.920000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.973134 | 0.960000 | 0.960000 | 0.960000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.986072 | 0.984000 | 0.984000 | 0.984000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.993617 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.997798 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 98 | 0.330080 | 0.989796 | 0.000000 | 0.329932 | 0.989796 | 2.979592 |
| right | 78 | 0.936256 | 0.961538 | 0.961538 | 0.961538 | 0.000000 | 1.000000 |
| left | 102 | 0.855177 | 0.803922 | 0.803922 | 0.803922 | 0.000000 | 1.000000 |
| forward | 117 | 0.899840 | 0.914530 | 0.914530 | 0.914530 | 0.000000 | 1.000000 |
| backward | 105 | 0.893859 | 0.780952 | 0.780952 | 0.780952 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.288884 | 0.333333 | 0.044449 |
| [0.3,0.4] | 101 | 0.331499 | 0.346535 | 0.015036 |
| [0.4,0.5] | 16 | 0.452262 | 0.500000 | 0.047738 |
| [0.5,0.6] | 14 | 0.556550 | 0.571429 | 0.014878 |
| [0.6,0.7] | 18 | 0.664537 | 0.277778 | 0.386759 |
| [0.7,0.8] | 29 | 0.747312 | 0.689655 | 0.057657 |
| [0.8,0.9] | 36 | 0.847805 | 0.694444 | 0.153360 |
| [0.9,1.0] | 285 | 0.978202 | 0.971930 | 0.006272 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

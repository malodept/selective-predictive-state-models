# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.779447 |
| biased top-1 | 0.880000 |
| strict top-1 | 0.698000 |
| tie-aware top-1 | 0.758667 |
| correct tied with another candidate | 0.182000 |
| mean tie count | 1.364000 |
| ECE vs tie-aware target | 0.024228 |
| ECE vs strict target | 0.084565 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.779447 | 0.880000 | 0.698000 | 0.758667 | 0.182000 | 1.364000 |
| 0.90 | 450 | 0.829901 | 0.868889 | 0.775556 | 0.806667 | 0.093333 | 1.186667 |
| 0.80 | 400 | 0.890867 | 0.867500 | 0.867500 | 0.867500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.937030 | 0.920000 | 0.920000 | 0.920000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.967693 | 0.956667 | 0.956667 | 0.956667 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.985810 | 0.980000 | 0.980000 | 0.980000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.994529 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.997970 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 91 | 0.328894 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 99 | 0.853807 | 0.808081 | 0.808081 | 0.808081 | 0.000000 | 1.000000 |
| left | 120 | 0.890839 | 0.841667 | 0.841667 | 0.841667 | 0.000000 | 1.000000 |
| forward | 97 | 0.873445 | 0.824742 | 0.824742 | 0.824742 | 0.000000 | 1.000000 |
| backward | 93 | 0.899382 | 0.946237 | 0.946237 | 0.946237 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 2 | 0.292177 | 0.333333 | 0.041156 |
| [0.3,0.4] | 96 | 0.333008 | 0.319444 | 0.013564 |
| [0.4,0.5] | 12 | 0.457560 | 0.416667 | 0.040893 |
| [0.5,0.6] | 24 | 0.563510 | 0.458333 | 0.105176 |
| [0.6,0.7] | 24 | 0.652319 | 0.666667 | 0.014347 |
| [0.7,0.8] | 34 | 0.756686 | 0.617647 | 0.139039 |
| [0.8,0.9] | 44 | 0.853746 | 0.863636 | 0.009891 |
| [0.9,1.0] | 264 | 0.981847 | 0.973485 | 0.008362 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

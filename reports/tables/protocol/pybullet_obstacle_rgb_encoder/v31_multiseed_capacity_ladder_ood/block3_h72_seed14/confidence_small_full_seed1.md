# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.779383 |
| biased top-1 | 0.876000 |
| strict top-1 | 0.702000 |
| tie-aware top-1 | 0.760000 |
| correct tied with another candidate | 0.174000 |
| mean tie count | 1.348000 |
| ECE vs tie-aware target | 0.038259 |
| ECE vs strict target | 0.085071 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.779383 | 0.876000 | 0.702000 | 0.760000 | 0.174000 | 1.348000 |
| 0.90 | 450 | 0.829694 | 0.862222 | 0.780000 | 0.807407 | 0.082222 | 1.164444 |
| 0.80 | 400 | 0.889832 | 0.857500 | 0.857500 | 0.857500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.937068 | 0.914286 | 0.914286 | 0.914286 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.970490 | 0.950000 | 0.950000 | 0.950000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.988027 | 0.976000 | 0.976000 | 0.976000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.994823 | 0.990000 | 0.990000 | 0.990000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998144 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 87 | 0.329181 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 107 | 0.888064 | 0.887850 | 0.887850 | 0.887850 | 0.000000 | 1.000000 |
| left | 102 | 0.815162 | 0.754902 | 0.754902 | 0.754902 | 0.000000 | 1.000000 |
| forward | 106 | 0.910355 | 0.877358 | 0.877358 | 0.877358 | 0.000000 | 1.000000 |
| backward | 98 | 0.881486 | 0.877551 | 0.877551 | 0.877551 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 2 | 0.292721 | 0.333333 | 0.040613 |
| [0.3,0.4] | 92 | 0.332799 | 0.362319 | 0.029520 |
| [0.4,0.5] | 12 | 0.450871 | 0.583333 | 0.132462 |
| [0.5,0.6] | 32 | 0.546907 | 0.468750 | 0.078157 |
| [0.6,0.7] | 28 | 0.660279 | 0.500000 | 0.160279 |
| [0.7,0.8] | 25 | 0.746700 | 0.760000 | 0.013300 |
| [0.8,0.9] | 39 | 0.845424 | 0.743590 | 0.101834 |
| [0.9,1.0] | 270 | 0.983149 | 0.970370 | 0.012778 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

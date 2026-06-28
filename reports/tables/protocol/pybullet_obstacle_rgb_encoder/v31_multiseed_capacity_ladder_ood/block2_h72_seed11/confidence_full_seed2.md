# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.862008 |
| biased top-1 | 0.984000 |
| strict top-1 | 0.814000 |
| tie-aware top-1 | 0.871000 |
| correct tied with another candidate | 0.170000 |
| mean tie count | 1.338000 |
| ECE vs tie-aware target | 0.011674 |
| ECE vs strict target | 0.066259 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.862008 | 0.984000 | 0.814000 | 0.871000 | 0.168000 | 1.338000 |
| 0.90 | 450 | 0.921114 | 0.984444 | 0.904444 | 0.931481 | 0.077778 | 1.157778 |
| 0.80 | 400 | 0.986054 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.995973 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.998446 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999289 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999675 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999861 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 84 | 0.331486 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 97 | 0.953622 | 0.969072 | 0.969072 | 0.969072 | 0.000000 | 1.000000 |
| left | 100 | 0.945887 | 0.950000 | 0.950000 | 0.950000 | 0.000000 | 1.000000 |
| forward | 114 | 0.991759 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| backward | 105 | 0.981034 | 1.000000 | 0.990476 | 0.995238 | 0.009524 | 1.009524 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 87 | 0.332142 | 0.333333 | 0.001191 |
| [0.4,0.5] | 2 | 0.462108 | 0.750000 | 0.287892 |
| [0.5,0.6] | 5 | 0.534080 | 0.400000 | 0.134080 |
| [0.6,0.7] | 4 | 0.636588 | 0.750000 | 0.113412 |
| [0.7,0.8] | 6 | 0.785335 | 0.833333 | 0.047999 |
| [0.8,0.9] | 9 | 0.857789 | 1.000000 | 0.142211 |
| [0.9,1.0] | 387 | 0.991045 | 0.997416 | 0.006371 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

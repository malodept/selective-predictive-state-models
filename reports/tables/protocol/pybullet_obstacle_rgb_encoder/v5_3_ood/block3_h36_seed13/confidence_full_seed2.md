# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.801318 |
| biased top-1 | 0.886000 |
| strict top-1 | 0.704000 |
| tie-aware top-1 | 0.764667 |
| correct tied with another candidate | 0.182000 |
| mean tie count | 1.364000 |
| ECE vs tie-aware target | 0.038089 |
| ECE vs strict target | 0.097318 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.801318 | 0.886000 | 0.704000 | 0.764667 | 0.182000 | 1.364000 |
| 0.90 | 450 | 0.854035 | 0.873333 | 0.782222 | 0.812593 | 0.091111 | 1.182222 |
| 0.80 | 400 | 0.916741 | 0.872500 | 0.872500 | 0.872500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.962659 | 0.922857 | 0.922857 | 0.922857 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.986916 | 0.983333 | 0.983333 | 0.983333 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.994890 | 0.996000 | 0.996000 | 0.996000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.997924 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999221 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 91 | 0.329383 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 99 | 0.846550 | 0.757576 | 0.757576 | 0.757576 | 0.000000 | 1.000000 |
| left | 120 | 0.904299 | 0.841667 | 0.841667 | 0.841667 | 0.000000 | 1.000000 |
| forward | 97 | 0.926063 | 0.896907 | 0.896907 | 0.896907 | 0.000000 | 1.000000 |
| backward | 93 | 0.951966 | 0.956989 | 0.956989 | 0.956989 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 91 | 0.329383 | 0.333333 | 0.003950 |
| [0.4,0.5] | 12 | 0.455430 | 0.333333 | 0.122096 |
| [0.5,0.6] | 23 | 0.549406 | 0.521739 | 0.027667 |
| [0.6,0.7] | 22 | 0.647832 | 0.590909 | 0.056923 |
| [0.7,0.8] | 24 | 0.754989 | 0.416667 | 0.338322 |
| [0.8,0.9] | 27 | 0.860549 | 0.629630 | 0.230920 |
| [0.9,1.0] | 301 | 0.986634 | 0.983389 | 0.003245 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

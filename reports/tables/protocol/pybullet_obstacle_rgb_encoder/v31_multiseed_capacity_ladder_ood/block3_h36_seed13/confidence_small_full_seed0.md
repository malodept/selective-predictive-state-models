# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.781383 |
| biased top-1 | 0.916000 |
| strict top-1 | 0.734000 |
| tie-aware top-1 | 0.794667 |
| correct tied with another candidate | 0.182000 |
| mean tie count | 1.364000 |
| ECE vs tie-aware target | 0.016722 |
| ECE vs strict target | 0.069027 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.781383 | 0.916000 | 0.734000 | 0.794667 | 0.182000 | 1.364000 |
| 0.90 | 450 | 0.831891 | 0.906667 | 0.815556 | 0.845926 | 0.091111 | 1.182222 |
| 0.80 | 400 | 0.892774 | 0.905000 | 0.905000 | 0.905000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.939742 | 0.945714 | 0.945714 | 0.945714 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.970815 | 0.983333 | 0.983333 | 0.983333 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.986447 | 0.988000 | 0.988000 | 0.988000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.994280 | 0.990000 | 0.990000 | 0.990000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.997854 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 91 | 0.329433 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 99 | 0.844338 | 0.858586 | 0.858586 | 0.858586 | 0.000000 | 1.000000 |
| left | 120 | 0.846014 | 0.866667 | 0.866667 | 0.866667 | 0.000000 | 1.000000 |
| forward | 97 | 0.912748 | 0.907216 | 0.907216 | 0.907216 | 0.000000 | 1.000000 |
| backward | 93 | 0.936189 | 0.967742 | 0.967742 | 0.967742 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.292127 | 0.333333 | 0.041206 |
| [0.3,0.4] | 96 | 0.332822 | 0.354167 | 0.021344 |
| [0.4,0.5] | 12 | 0.466098 | 0.416667 | 0.049431 |
| [0.5,0.6] | 25 | 0.555951 | 0.600000 | 0.044049 |
| [0.6,0.7] | 24 | 0.643126 | 0.708333 | 0.065208 |
| [0.7,0.8] | 32 | 0.752613 | 0.781250 | 0.028637 |
| [0.8,0.9] | 42 | 0.863485 | 0.857143 | 0.006343 |
| [0.9,1.0] | 268 | 0.981984 | 0.988806 | 0.006822 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

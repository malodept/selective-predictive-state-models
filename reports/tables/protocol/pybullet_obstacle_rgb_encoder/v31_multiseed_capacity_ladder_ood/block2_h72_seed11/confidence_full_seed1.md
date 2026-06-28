# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.859501 |
| biased top-1 | 0.974000 |
| strict top-1 | 0.804000 |
| tie-aware top-1 | 0.861000 |
| correct tied with another candidate | 0.170000 |
| mean tie count | 1.338000 |
| ECE vs tie-aware target | 0.016884 |
| ECE vs strict target | 0.073300 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.859501 | 0.974000 | 0.804000 | 0.861000 | 0.168000 | 1.338000 |
| 0.90 | 450 | 0.918281 | 0.971111 | 0.893333 | 0.919630 | 0.075556 | 1.153333 |
| 0.80 | 400 | 0.983562 | 0.992500 | 0.992500 | 0.992500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.996566 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.998679 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999401 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999742 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999882 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 84 | 0.331595 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 97 | 0.967495 | 0.969072 | 0.969072 | 0.969072 | 0.000000 | 1.000000 |
| left | 100 | 0.949443 | 0.940000 | 0.940000 | 0.940000 | 0.000000 | 1.000000 |
| forward | 114 | 0.967667 | 0.964912 | 0.964912 | 0.964912 | 0.000000 | 1.000000 |
| backward | 105 | 0.978966 | 1.000000 | 0.990476 | 0.995238 | 0.009524 | 1.009524 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 84 | 0.331595 | 0.333333 | 0.001738 |
| [0.4,0.5] | 4 | 0.445777 | 0.375000 | 0.070777 |
| [0.5,0.6] | 10 | 0.541490 | 0.300000 | 0.241490 |
| [0.6,0.7] | 2 | 0.636992 | 0.500000 | 0.136992 |
| [0.7,0.8] | 9 | 0.763829 | 0.666667 | 0.097162 |
| [0.8,0.9] | 11 | 0.861269 | 1.000000 | 0.138731 |
| [0.9,1.0] | 380 | 0.992306 | 1.000000 | 0.007694 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

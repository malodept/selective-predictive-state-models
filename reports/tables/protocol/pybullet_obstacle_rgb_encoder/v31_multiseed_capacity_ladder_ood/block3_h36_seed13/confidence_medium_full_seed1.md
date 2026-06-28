# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.806910 |
| biased top-1 | 0.910000 |
| strict top-1 | 0.728000 |
| tie-aware top-1 | 0.788667 |
| correct tied with another candidate | 0.182000 |
| mean tie count | 1.364000 |
| ECE vs tie-aware target | 0.021923 |
| ECE vs strict target | 0.082430 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.806910 | 0.910000 | 0.728000 | 0.788667 | 0.182000 | 1.364000 |
| 0.90 | 450 | 0.860272 | 0.900000 | 0.808889 | 0.839259 | 0.091111 | 1.182222 |
| 0.80 | 400 | 0.924347 | 0.902500 | 0.902500 | 0.902500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.966041 | 0.951429 | 0.951429 | 0.951429 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.985049 | 0.976667 | 0.976667 | 0.976667 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.993233 | 0.988000 | 0.988000 | 0.988000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.997065 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998872 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 91 | 0.329319 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 99 | 0.869040 | 0.838384 | 0.838384 | 0.838384 | 0.000000 | 1.000000 |
| left | 120 | 0.900720 | 0.858333 | 0.858333 | 0.858333 | 0.000000 | 1.000000 |
| forward | 97 | 0.923922 | 0.896907 | 0.896907 | 0.896907 | 0.000000 | 1.000000 |
| backward | 93 | 0.965004 | 0.978495 | 0.978495 | 0.978495 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.293322 | 0.333333 | 0.040012 |
| [0.3,0.4] | 93 | 0.330856 | 0.322581 | 0.008275 |
| [0.4,0.5] | 8 | 0.455423 | 0.375000 | 0.080423 |
| [0.5,0.6] | 20 | 0.556002 | 0.600000 | 0.043998 |
| [0.6,0.7] | 14 | 0.657494 | 0.642857 | 0.014636 |
| [0.7,0.8] | 21 | 0.753550 | 0.571429 | 0.182121 |
| [0.8,0.9] | 42 | 0.861379 | 0.809524 | 0.051855 |
| [0.9,1.0] | 301 | 0.984789 | 0.976744 | 0.008045 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

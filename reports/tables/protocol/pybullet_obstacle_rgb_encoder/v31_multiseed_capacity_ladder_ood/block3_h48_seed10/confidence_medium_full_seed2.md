# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.790830 |
| biased top-1 | 0.904000 |
| strict top-1 | 0.710000 |
| tie-aware top-1 | 0.774667 |
| correct tied with another candidate | 0.194000 |
| mean tie count | 1.388000 |
| ECE vs tie-aware target | 0.026983 |
| ECE vs strict target | 0.088130 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.790830 | 0.904000 | 0.710000 | 0.774667 | 0.196000 | 1.388000 |
| 0.90 | 450 | 0.842416 | 0.895556 | 0.788889 | 0.824444 | 0.106667 | 1.213333 |
| 0.80 | 400 | 0.905806 | 0.885000 | 0.885000 | 0.885000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.951003 | 0.931429 | 0.931429 | 0.931429 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.975031 | 0.956667 | 0.956667 | 0.956667 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.987494 | 0.976000 | 0.976000 | 0.976000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.994565 | 0.985000 | 0.985000 | 0.985000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.997994 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 98 | 0.329597 | 0.989796 | 0.000000 | 0.329932 | 0.989796 | 2.979592 |
| right | 78 | 0.921117 | 0.923077 | 0.923077 | 0.923077 | 0.000000 | 1.000000 |
| left | 102 | 0.864898 | 0.872549 | 0.872549 | 0.872549 | 0.000000 | 1.000000 |
| forward | 117 | 0.924833 | 0.905983 | 0.905983 | 0.905983 | 0.000000 | 1.000000 |
| backward | 105 | 0.903259 | 0.838095 | 0.838095 | 0.838095 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 2 | 0.285832 | 0.166667 | 0.119165 |
| [0.3,0.4] | 97 | 0.331134 | 0.340206 | 0.009072 |
| [0.4,0.5] | 10 | 0.451625 | 0.600000 | 0.148375 |
| [0.5,0.6] | 19 | 0.556494 | 0.473684 | 0.082810 |
| [0.6,0.7] | 17 | 0.660354 | 0.529412 | 0.130942 |
| [0.7,0.8] | 25 | 0.746353 | 0.760000 | 0.013647 |
| [0.8,0.9] | 47 | 0.857587 | 0.808511 | 0.049077 |
| [0.9,1.0] | 283 | 0.980361 | 0.964664 | 0.015697 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

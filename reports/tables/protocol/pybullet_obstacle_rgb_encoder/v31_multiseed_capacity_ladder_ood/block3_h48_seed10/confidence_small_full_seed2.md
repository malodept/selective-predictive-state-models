# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.789597 |
| biased top-1 | 0.888000 |
| strict top-1 | 0.694000 |
| tie-aware top-1 | 0.758667 |
| correct tied with another candidate | 0.194000 |
| mean tie count | 1.388000 |
| ECE vs tie-aware target | 0.035936 |
| ECE vs strict target | 0.097157 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.789597 | 0.888000 | 0.694000 | 0.758667 | 0.196000 | 1.388000 |
| 0.90 | 450 | 0.840920 | 0.877778 | 0.771111 | 0.806667 | 0.106667 | 1.213333 |
| 0.80 | 400 | 0.904290 | 0.865000 | 0.865000 | 0.865000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.951682 | 0.934286 | 0.934286 | 0.934286 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.976282 | 0.976667 | 0.976667 | 0.976667 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.989289 | 0.996000 | 0.996000 | 0.996000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.995734 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998366 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 98 | 0.330186 | 0.989796 | 0.000000 | 0.329932 | 0.989796 | 2.979592 |
| right | 78 | 0.892157 | 0.884615 | 0.884615 | 0.884615 | 0.000000 | 1.000000 |
| left | 102 | 0.876861 | 0.843137 | 0.843137 | 0.843137 | 0.000000 | 1.000000 |
| forward | 117 | 0.910082 | 0.905983 | 0.905983 | 0.905983 | 0.000000 | 1.000000 |
| backward | 105 | 0.923169 | 0.819048 | 0.819048 | 0.819048 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 101 | 0.331404 | 0.339934 | 0.008530 |
| [0.4,0.5] | 5 | 0.451969 | 0.200000 | 0.251969 |
| [0.5,0.6] | 28 | 0.553941 | 0.285714 | 0.268227 |
| [0.6,0.7] | 18 | 0.659039 | 0.500000 | 0.159039 |
| [0.7,0.8] | 17 | 0.750236 | 0.588235 | 0.162001 |
| [0.8,0.9] | 46 | 0.854996 | 0.804348 | 0.050648 |
| [0.9,1.0] | 285 | 0.981088 | 0.982456 | 0.001368 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

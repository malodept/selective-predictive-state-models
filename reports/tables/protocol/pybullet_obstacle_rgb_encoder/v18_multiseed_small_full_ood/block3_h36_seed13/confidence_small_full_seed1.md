# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.784989 |
| biased top-1 | 0.878000 |
| strict top-1 | 0.696000 |
| tie-aware top-1 | 0.756667 |
| correct tied with another candidate | 0.182000 |
| mean tie count | 1.364000 |
| ECE vs tie-aware target | 0.035773 |
| ECE vs strict target | 0.096282 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.784989 | 0.878000 | 0.696000 | 0.756667 | 0.182000 | 1.364000 |
| 0.90 | 450 | 0.835950 | 0.864444 | 0.773333 | 0.803704 | 0.091111 | 1.182222 |
| 0.80 | 400 | 0.897535 | 0.865000 | 0.865000 | 0.865000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.944639 | 0.931429 | 0.931429 | 0.931429 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.975257 | 0.970000 | 0.970000 | 0.970000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.989001 | 0.996000 | 0.996000 | 0.996000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.995061 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998348 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 91 | 0.329100 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 99 | 0.838412 | 0.818182 | 0.818182 | 0.818182 | 0.000000 | 1.000000 |
| left | 120 | 0.870326 | 0.791667 | 0.791667 | 0.791667 | 0.000000 | 1.000000 |
| forward | 97 | 0.925593 | 0.896907 | 0.896907 | 0.896907 | 0.000000 | 1.000000 |
| backward | 93 | 0.917439 | 0.913978 | 0.913978 | 0.913978 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.293973 | 0.333333 | 0.039360 |
| [0.3,0.4] | 95 | 0.331663 | 0.326316 | 0.005347 |
| [0.4,0.5] | 11 | 0.456188 | 0.272727 | 0.183461 |
| [0.5,0.6] | 28 | 0.549512 | 0.464286 | 0.085226 |
| [0.6,0.7] | 22 | 0.659277 | 0.454545 | 0.204732 |
| [0.7,0.8] | 27 | 0.745964 | 0.703704 | 0.042260 |
| [0.8,0.9] | 37 | 0.850435 | 0.702703 | 0.147732 |
| [0.9,1.0] | 279 | 0.982712 | 0.989247 | 0.006535 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

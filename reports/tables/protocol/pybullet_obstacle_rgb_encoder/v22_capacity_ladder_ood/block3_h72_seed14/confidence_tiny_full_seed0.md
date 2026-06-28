# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.756339 |
| biased top-1 | 0.848000 |
| strict top-1 | 0.674000 |
| tie-aware top-1 | 0.732000 |
| correct tied with another candidate | 0.174000 |
| mean tie count | 1.348000 |
| ECE vs tie-aware target | 0.045441 |
| ECE vs strict target | 0.095571 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.756339 | 0.848000 | 0.674000 | 0.732000 | 0.174000 | 1.348000 |
| 0.90 | 450 | 0.804081 | 0.833333 | 0.742222 | 0.772593 | 0.091111 | 1.182222 |
| 0.80 | 400 | 0.861632 | 0.825000 | 0.825000 | 0.825000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.908541 | 0.885714 | 0.885714 | 0.885714 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.946915 | 0.946667 | 0.946667 | 0.946667 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.975175 | 0.984000 | 0.984000 | 0.984000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.990283 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.996508 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 87 | 0.330031 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 107 | 0.803695 | 0.803738 | 0.803738 | 0.803738 | 0.000000 | 1.000000 |
| left | 102 | 0.829827 | 0.774510 | 0.774510 | 0.774510 | 0.000000 | 1.000000 |
| forward | 106 | 0.844090 | 0.801887 | 0.801887 | 0.801887 | 0.000000 | 1.000000 |
| backward | 98 | 0.911687 | 0.887755 | 0.887755 | 0.887755 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.298377 | 0.333333 | 0.034956 |
| [0.3,0.4] | 93 | 0.330474 | 0.351254 | 0.020780 |
| [0.4,0.5] | 17 | 0.445609 | 0.294118 | 0.151492 |
| [0.5,0.6] | 39 | 0.553132 | 0.461538 | 0.091594 |
| [0.6,0.7] | 33 | 0.652187 | 0.424242 | 0.227945 |
| [0.7,0.8] | 42 | 0.757100 | 0.714286 | 0.042814 |
| [0.8,0.9] | 42 | 0.856599 | 0.809524 | 0.047075 |
| [0.9,1.0] | 233 | 0.981510 | 0.995708 | 0.014198 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

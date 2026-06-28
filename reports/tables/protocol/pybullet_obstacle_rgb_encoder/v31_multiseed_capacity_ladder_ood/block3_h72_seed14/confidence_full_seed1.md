# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.783400 |
| biased top-1 | 0.842000 |
| strict top-1 | 0.668000 |
| tie-aware top-1 | 0.726000 |
| correct tied with another candidate | 0.174000 |
| mean tie count | 1.348000 |
| ECE vs tie-aware target | 0.062904 |
| ECE vs strict target | 0.115400 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.783400 | 0.842000 | 0.668000 | 0.726000 | 0.174000 | 1.348000 |
| 0.90 | 450 | 0.833960 | 0.824444 | 0.742222 | 0.769630 | 0.082222 | 1.164444 |
| 0.80 | 400 | 0.893897 | 0.815000 | 0.815000 | 0.815000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.939402 | 0.871429 | 0.871429 | 0.871429 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.970897 | 0.916667 | 0.916667 | 0.916667 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.986969 | 0.952000 | 0.952000 | 0.952000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.995073 | 0.970000 | 0.970000 | 0.970000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998448 | 0.986667 | 0.986667 | 0.986667 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 87 | 0.330210 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 107 | 0.900214 | 0.850467 | 0.850467 | 0.850467 | 0.000000 | 1.000000 |
| left | 102 | 0.830161 | 0.784314 | 0.784314 | 0.784314 | 0.000000 | 1.000000 |
| forward | 106 | 0.881802 | 0.754717 | 0.754717 | 0.754717 | 0.000000 | 1.000000 |
| backward | 98 | 0.903075 | 0.846939 | 0.846939 | 0.846939 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 92 | 0.332869 | 0.347826 | 0.014957 |
| [0.4,0.5] | 17 | 0.463355 | 0.411765 | 0.051590 |
| [0.5,0.6] | 24 | 0.558849 | 0.458333 | 0.100516 |
| [0.6,0.7] | 29 | 0.662938 | 0.379310 | 0.283627 |
| [0.7,0.8] | 25 | 0.750093 | 0.720000 | 0.030093 |
| [0.8,0.9] | 36 | 0.841213 | 0.777778 | 0.063436 |
| [0.9,1.0] | 277 | 0.980236 | 0.924188 | 0.056048 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

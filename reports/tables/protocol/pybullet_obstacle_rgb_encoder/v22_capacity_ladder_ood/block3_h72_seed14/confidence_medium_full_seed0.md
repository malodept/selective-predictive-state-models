# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.784829 |
| biased top-1 | 0.870000 |
| strict top-1 | 0.696000 |
| tie-aware top-1 | 0.754000 |
| correct tied with another candidate | 0.174000 |
| mean tie count | 1.348000 |
| ECE vs tie-aware target | 0.043887 |
| ECE vs strict target | 0.101887 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.784829 | 0.870000 | 0.696000 | 0.754000 | 0.174000 | 1.348000 |
| 0.90 | 450 | 0.835585 | 0.857778 | 0.773333 | 0.801481 | 0.084444 | 1.168889 |
| 0.80 | 400 | 0.895104 | 0.857500 | 0.857500 | 0.857500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.940897 | 0.902857 | 0.902857 | 0.902857 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.968396 | 0.946667 | 0.946667 | 0.946667 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.985004 | 0.964000 | 0.964000 | 0.964000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.993453 | 0.975000 | 0.975000 | 0.975000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.997789 | 0.986667 | 0.986667 | 0.986667 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 87 | 0.330333 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 107 | 0.887173 | 0.841121 | 0.841121 | 0.841121 | 0.000000 | 1.000000 |
| left | 102 | 0.835427 | 0.784314 | 0.784314 | 0.784314 | 0.000000 | 1.000000 |
| forward | 106 | 0.894848 | 0.830189 | 0.830189 | 0.830189 | 0.000000 | 1.000000 |
| backward | 98 | 0.904904 | 0.918367 | 0.918367 | 0.918367 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 88 | 0.330074 | 0.329545 | 0.000528 |
| [0.4,0.5] | 16 | 0.454673 | 0.437500 | 0.017173 |
| [0.5,0.6] | 29 | 0.544878 | 0.413793 | 0.131085 |
| [0.6,0.7] | 21 | 0.654076 | 0.809524 | 0.155448 |
| [0.7,0.8] | 26 | 0.753539 | 0.538462 | 0.215077 |
| [0.8,0.9] | 49 | 0.847336 | 0.795918 | 0.051417 |
| [0.9,1.0] | 271 | 0.979501 | 0.955720 | 0.023781 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.852438 |
| biased top-1 | 0.976000 |
| strict top-1 | 0.806000 |
| tie-aware top-1 | 0.863000 |
| correct tied with another candidate | 0.170000 |
| mean tie count | 1.338000 |
| ECE vs tie-aware target | 0.013139 |
| ECE vs strict target | 0.067507 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.852438 | 0.976000 | 0.806000 | 0.863000 | 0.168000 | 1.338000 |
| 0.90 | 450 | 0.910459 | 0.973333 | 0.895556 | 0.921852 | 0.075556 | 1.153333 |
| 0.80 | 400 | 0.972639 | 0.982500 | 0.982500 | 0.982500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.991770 | 0.994286 | 0.994286 | 0.994286 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.996540 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.998380 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999355 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999761 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 84 | 0.331453 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 97 | 0.941679 | 0.948454 | 0.948454 | 0.948454 | 0.000000 | 1.000000 |
| left | 100 | 0.927453 | 0.930000 | 0.930000 | 0.930000 | 0.000000 | 1.000000 |
| forward | 114 | 0.978239 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| backward | 105 | 0.978757 | 1.000000 | 0.990476 | 0.995238 | 0.009524 | 1.009524 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 84 | 0.331453 | 0.333333 | 0.001880 |
| [0.4,0.5] | 2 | 0.478196 | 0.750000 | 0.271804 |
| [0.5,0.6] | 6 | 0.537911 | 0.500000 | 0.037911 |
| [0.6,0.7] | 9 | 0.648212 | 0.777778 | 0.129566 |
| [0.7,0.8] | 14 | 0.744049 | 0.714286 | 0.029763 |
| [0.8,0.9] | 21 | 0.853683 | 0.952381 | 0.098698 |
| [0.9,1.0] | 364 | 0.989053 | 0.994505 | 0.005453 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

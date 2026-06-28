# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.775080 |
| biased top-1 | 0.866000 |
| strict top-1 | 0.692000 |
| tie-aware top-1 | 0.750000 |
| correct tied with another candidate | 0.174000 |
| mean tie count | 1.348000 |
| ECE vs tie-aware target | 0.029789 |
| ECE vs strict target | 0.087641 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.775080 | 0.866000 | 0.692000 | 0.750000 | 0.174000 | 1.348000 |
| 0.90 | 450 | 0.824866 | 0.853333 | 0.768889 | 0.797037 | 0.084444 | 1.168889 |
| 0.80 | 400 | 0.882787 | 0.847500 | 0.847500 | 0.847500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.927820 | 0.902857 | 0.902857 | 0.902857 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.960888 | 0.950000 | 0.950000 | 0.950000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.981352 | 0.972000 | 0.972000 | 0.972000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.992238 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.997163 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 87 | 0.329590 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 107 | 0.889388 | 0.878505 | 0.878505 | 0.878505 | 0.000000 | 1.000000 |
| left | 102 | 0.785242 | 0.696078 | 0.696078 | 0.696078 | 0.000000 | 1.000000 |
| forward | 106 | 0.889845 | 0.858491 | 0.858491 | 0.858491 | 0.000000 | 1.000000 |
| backward | 98 | 0.911051 | 0.918367 | 0.918367 | 0.918367 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.296292 | 0.333333 | 0.037041 |
| [0.3,0.4] | 87 | 0.329840 | 0.329502 | 0.000338 |
| [0.4,0.5] | 19 | 0.466442 | 0.421053 | 0.045389 |
| [0.5,0.6] | 27 | 0.550371 | 0.370370 | 0.180001 |
| [0.6,0.7] | 26 | 0.648459 | 0.692308 | 0.043849 |
| [0.7,0.8] | 41 | 0.743857 | 0.634146 | 0.109710 |
| [0.8,0.9] | 46 | 0.856962 | 0.826087 | 0.030875 |
| [0.9,1.0] | 253 | 0.980423 | 0.972332 | 0.008091 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

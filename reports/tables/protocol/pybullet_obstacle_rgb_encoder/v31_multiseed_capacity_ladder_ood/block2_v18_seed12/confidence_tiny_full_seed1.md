# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/tiny_full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.843892 |
| biased top-1 | 1.000000 |
| strict top-1 | 0.788000 |
| tie-aware top-1 | 0.858667 |
| correct tied with another candidate | 0.212000 |
| mean tie count | 1.424000 |
| ECE vs tie-aware target | 0.014775 |
| ECE vs strict target | 0.083891 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.843892 | 1.000000 | 0.788000 | 0.858667 | 0.212000 | 1.424000 |
| 0.90 | 450 | 0.901443 | 1.000000 | 0.875556 | 0.917037 | 0.124444 | 1.248889 |
| 0.80 | 400 | 0.972500 | 1.000000 | 0.985000 | 0.990000 | 0.015000 | 1.030000 |
| 0.70 | 350 | 0.997776 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.999144 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.999595 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.999823 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.999937 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 106 | 0.329677 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 113 | 0.985829 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| left | 105 | 0.969780 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| forward | 83 | 0.984326 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |
| backward | 93 | 0.990058 | 1.000000 | 1.000000 | 1.000000 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 2 | 0.293988 | 0.333333 | 0.039346 |
| [0.3,0.4] | 104 | 0.330363 | 0.333333 | 0.002970 |
| [0.4,0.5] | 3 | 0.428880 | 1.000000 | 0.571120 |
| [0.5,0.6] | 1 | 0.505231 | 1.000000 | 0.494769 |
| [0.6,0.7] | 3 | 0.661619 | 1.000000 | 0.338381 |
| [0.7,0.8] | 3 | 0.760895 | 1.000000 | 0.239105 |
| [0.8,0.9] | 6 | 0.850379 | 1.000000 | 0.149621 |
| [0.9,1.0] | 378 | 0.994282 | 1.000000 | 0.005718 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

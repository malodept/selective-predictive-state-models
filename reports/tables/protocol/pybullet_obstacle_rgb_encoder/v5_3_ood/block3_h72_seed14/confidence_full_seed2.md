# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed2/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.780608 |
| biased top-1 | 0.832000 |
| strict top-1 | 0.658000 |
| tie-aware top-1 | 0.716000 |
| correct tied with another candidate | 0.174000 |
| mean tie count | 1.348000 |
| ECE vs tie-aware target | 0.064608 |
| ECE vs strict target | 0.122608 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.780608 | 0.832000 | 0.658000 | 0.716000 | 0.174000 | 1.348000 |
| 0.90 | 450 | 0.830966 | 0.820000 | 0.731111 | 0.760741 | 0.088889 | 1.177778 |
| 0.80 | 400 | 0.891137 | 0.815000 | 0.815000 | 0.815000 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.937873 | 0.877143 | 0.877143 | 0.877143 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.970745 | 0.943333 | 0.943333 | 0.943333 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.989218 | 0.988000 | 0.988000 | 0.988000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.996388 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998998 | 0.993333 | 0.993333 | 0.993333 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 87 | 0.330316 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 107 | 0.855433 | 0.663551 | 0.663551 | 0.663551 | 0.000000 | 1.000000 |
| left | 102 | 0.827891 | 0.784314 | 0.784314 | 0.784314 | 0.000000 | 1.000000 |
| forward | 106 | 0.893754 | 0.830189 | 0.830189 | 0.830189 | 0.000000 | 1.000000 |
| backward | 98 | 0.927064 | 0.918367 | 0.918367 | 0.918367 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 1 | 0.293966 | 0.000000 | 0.293966 |
| [0.3,0.4] | 91 | 0.330503 | 0.329670 | 0.000833 |
| [0.4,0.5] | 12 | 0.451076 | 0.416667 | 0.034409 |
| [0.5,0.6] | 33 | 0.552528 | 0.333333 | 0.219195 |
| [0.6,0.7] | 27 | 0.640284 | 0.333333 | 0.306951 |
| [0.7,0.8] | 26 | 0.754059 | 0.500000 | 0.254059 |
| [0.8,0.9] | 45 | 0.853585 | 0.755556 | 0.098029 |
| [0.9,1.0] | 265 | 0.984842 | 0.966038 | 0.018805 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

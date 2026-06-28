# State-disjoint confidence diagnostic: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed1/checkpoint.pt`
- split source: `explicit`
- test groups: `500`
- temperature: `0.02`

## Global metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.797997 |
| biased top-1 | 0.898000 |
| strict top-1 | 0.716000 |
| tie-aware top-1 | 0.776667 |
| correct tied with another candidate | 0.182000 |
| mean tie count | 1.364000 |
| ECE vs tie-aware target | 0.027449 |
| ECE vs strict target | 0.083393 |

## Selective prediction curve

| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 500 | 0.797997 | 0.898000 | 0.716000 | 0.776667 | 0.182000 | 1.364000 |
| 0.90 | 450 | 0.850282 | 0.886667 | 0.795556 | 0.825926 | 0.091111 | 1.182222 |
| 0.80 | 400 | 0.913377 | 0.882500 | 0.882500 | 0.882500 | 0.000000 | 1.000000 |
| 0.70 | 350 | 0.959353 | 0.945714 | 0.945714 | 0.945714 | 0.000000 | 1.000000 |
| 0.60 | 300 | 0.980706 | 0.966667 | 0.966667 | 0.966667 | 0.000000 | 1.000000 |
| 0.50 | 250 | 0.991570 | 0.980000 | 0.980000 | 0.980000 | 0.000000 | 1.000000 |
| 0.40 | 200 | 0.996714 | 0.995000 | 0.995000 | 0.995000 | 0.000000 | 1.000000 |
| 0.30 | 150 | 0.998889 | 0.993333 | 0.993333 | 0.993333 | 0.000000 | 1.000000 |

## Action breakdown

| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stay | 91 | 0.329744 | 1.000000 | 0.000000 | 0.333333 | 1.000000 | 3.000000 |
| right | 99 | 0.871817 | 0.787879 | 0.787879 | 0.787879 | 0.000000 | 1.000000 |
| left | 120 | 0.902281 | 0.858333 | 0.858333 | 0.858333 | 0.000000 | 1.000000 |
| forward | 97 | 0.890390 | 0.886598 | 0.886598 | 0.886598 | 0.000000 | 1.000000 |
| backward | 93 | 0.946673 | 0.978495 | 0.978495 | 0.978495 | 0.000000 | 1.000000 |

## Calibration bins against tie-aware target

| confidence bin | count | mean confidence | mean target | gap |
| --- | ---: | ---: | ---: | ---: |
| [0.0,0.1] | 0 | NA | NA | NA |
| [0.1,0.2] | 0 | NA | NA | NA |
| [0.2,0.3] | 0 | NA | NA | NA |
| [0.3,0.4] | 94 | 0.331409 | 0.343972 | 0.012562 |
| [0.4,0.5] | 15 | 0.451102 | 0.333333 | 0.117768 |
| [0.5,0.6] | 18 | 0.553067 | 0.333333 | 0.219734 |
| [0.6,0.7] | 17 | 0.645370 | 0.529412 | 0.115958 |
| [0.7,0.8] | 20 | 0.759426 | 0.750000 | 0.009426 |
| [0.8,0.9] | 37 | 0.855436 | 0.864865 | 0.009429 |
| [0.9,1.0] | 299 | 0.980982 | 0.966555 | 0.014426 |

## Interpretation

This diagnostic asks whether model confidence tracks strict and tie-aware correctness.
A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.

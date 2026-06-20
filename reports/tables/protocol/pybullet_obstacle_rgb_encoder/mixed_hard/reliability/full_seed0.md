# Transformer reliability audit

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/delta_transformer/full_seed0/checkpoint.pt`
- mode: `full`
- test groups: `100`
- top-1 accuracy: `0.970000`
- mean confidence: `0.954140`
- ECE 10 bins: `0.024701`
- mean distance margin: `0.126243`

## Selective prediction curve

| coverage | kept groups | accuracy | mean confidence |
| ---: | ---: | ---: | ---: |
| 1.00 | 100 | 0.970000 | 0.954140 |
| 0.90 | 90 | 0.988889 | 0.985827 |
| 0.80 | 80 | 1.000000 | 0.995863 |
| 0.70 | 70 | 1.000000 | 0.997788 |
| 0.60 | 60 | 1.000000 | 0.998718 |
| 0.50 | 50 | 1.000000 | 0.999445 |
| 0.40 | 40 | 1.000000 | 0.999735 |
| 0.30 | 30 | 1.000000 | 0.999861 |

## Calibration bins

| confidence bin | count | accuracy | mean confidence |
| --- | ---: | ---: | ---: |
| [0.0, 0.1] | 0 | NA | NA |
| [0.1, 0.2] | 0 | NA | NA |
| [0.2, 0.3] | 0 | NA | NA |
| [0.3, 0.4] | 0 | NA | NA |
| [0.4, 0.5] | 0 | NA | NA |
| [0.5, 0.6] | 3 | 0.666667 | 0.544804 |
| [0.6, 0.7] | 3 | 0.666667 | 0.651949 |
| [0.7, 0.8] | 3 | 1.000000 | 0.760843 |
| [0.8, 0.9] | 6 | 1.000000 | 0.849864 |
| [0.9, 1.0] | 85 | 0.988235 | 0.993436 |

## Interpretation

This audit checks whether the model confidence is informative about correctness.
If high-confidence subsets have higher accuracy, the model can support selective prediction or selective compute.

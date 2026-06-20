# Transformer reliability audit

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/delta_transformer/full_seed1/checkpoint.pt`
- mode: `full`
- test groups: `100`
- top-1 accuracy: `0.940000`
- mean confidence: `0.907160`
- ECE 10 bins: `0.055080`
- mean distance margin: `0.098661`

## Selective prediction curve

| coverage | kept groups | accuracy | mean confidence |
| ---: | ---: | ---: | ---: |
| 1.00 | 100 | 0.940000 | 0.907160 |
| 0.90 | 90 | 0.977778 | 0.953812 |
| 0.80 | 80 | 1.000000 | 0.982530 |
| 0.70 | 70 | 1.000000 | 0.993282 |
| 0.60 | 60 | 1.000000 | 0.996694 |
| 0.50 | 50 | 1.000000 | 0.998107 |
| 0.40 | 40 | 1.000000 | 0.998651 |
| 0.30 | 30 | 1.000000 | 0.999189 |

## Calibration bins

| confidence bin | count | accuracy | mean confidence |
| --- | ---: | ---: | ---: |
| [0.0, 0.1] | 0 | NA | NA |
| [0.1, 0.2] | 0 | NA | NA |
| [0.2, 0.3] | 0 | NA | NA |
| [0.3, 0.4] | 1 | 1.000000 | 0.329103 |
| [0.4, 0.5] | 4 | 0.750000 | 0.451814 |
| [0.5, 0.6] | 4 | 0.250000 | 0.528003 |
| [0.6, 0.7] | 5 | 0.800000 | 0.650720 |
| [0.7, 0.8] | 5 | 0.800000 | 0.758938 |
| [0.8, 0.9] | 6 | 1.000000 | 0.867906 |
| [0.9, 1.0] | 75 | 1.000000 | 0.989492 |

## Interpretation

This audit checks whether the model confidence is informative about correctness.
If high-confidence subsets have higher accuracy, the model can support selective prediction or selective compute.

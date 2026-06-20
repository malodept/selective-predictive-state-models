# Transformer reliability audit

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/delta_transformer/full_seed2/checkpoint.pt`
- mode: `full`
- test groups: `100`
- top-1 accuracy: `0.950000`
- mean confidence: `0.949897`
- ECE 10 bins: `0.045876`
- mean distance margin: `0.128513`

## Selective prediction curve

| coverage | kept groups | accuracy | mean confidence |
| ---: | ---: | ---: | ---: |
| 1.00 | 100 | 0.950000 | 0.949897 |
| 0.90 | 90 | 1.000000 | 0.986234 |
| 0.80 | 80 | 1.000000 | 0.994321 |
| 0.70 | 70 | 1.000000 | 0.997515 |
| 0.60 | 60 | 1.000000 | 0.998912 |
| 0.50 | 50 | 1.000000 | 0.999502 |
| 0.40 | 40 | 1.000000 | 0.999778 |
| 0.30 | 30 | 1.000000 | 0.999896 |

## Calibration bins

| confidence bin | count | accuracy | mean confidence |
| --- | ---: | ---: | ---: |
| [0.0, 0.1] | 0 | NA | NA |
| [0.1, 0.2] | 0 | NA | NA |
| [0.2, 0.3] | 0 | NA | NA |
| [0.3, 0.4] | 1 | 1.000000 | 0.396273 |
| [0.4, 0.5] | 2 | 0.500000 | 0.439008 |
| [0.5, 0.6] | 2 | 0.000000 | 0.537097 |
| [0.6, 0.7] | 1 | 0.000000 | 0.690352 |
| [0.7, 0.8] | 2 | 0.500000 | 0.762068 |
| [0.8, 0.9] | 4 | 1.000000 | 0.849447 |
| [0.9, 1.0] | 88 | 1.000000 | 0.988965 |

## Interpretation

This audit checks whether the model confidence is informative about correctness.
If high-confidence subsets have higher accuracy, the model can support selective prediction or selective compute.

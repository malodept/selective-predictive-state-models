# Conditional residual shrinkage with affine validation calibration

Prediction family:

`z_pred = z_current + alpha(x) * delta_hat`

Global alpha selected on validation: `0.3600`.
Mixture coefficient selected on validation: `gamma=0.0600`.
Affine calibration selected on validation: `alpha_cal = clip(0.2800 + 0.1400 * alpha_raw, 0, 1)`.

| split | method | error | identity error | improvement vs identity | ratio vs identity | alpha mean | alpha std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val | identity | 1.432184 | 1.432184 | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| val | raw_residual_alpha_1 | 1.451741 | 1.432184 | -0.019557 | 1.013655 | 1.000000 | 0.000000 |
| val | global_alpha_val | 1.422708 | 1.432184 | 0.009477 | 0.993383 | 0.360000 | 0.000000 |
| val | conditional_alpha_raw | 1.428850 | 1.432184 | 0.003335 | 0.997672 | 0.589817 | 0.192880 |
| val | conditional_alpha_mixed_with_global | 1.422679 | 1.432184 | 0.009505 | 0.993363 | 0.373789 | 0.011573 |
| val | conditional_alpha_affine_calibrated | 1.422654 | 1.432184 | 0.009530 | 0.993346 | 0.362574 | 0.027003 |
| val | oracle_per_sample_alpha | 1.399817 | 1.432184 | 0.032368 | 0.977400 | 0.282404 | 0.351463 |
| test | identity | 1.606977 | 1.606977 | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| test | raw_residual_alpha_1 | 1.628799 | 1.606977 | -0.021823 | 1.013580 | 1.000000 | 0.000000 |
| test | global_alpha_val | 1.598825 | 1.606977 | 0.008152 | 0.994927 | 0.360000 | 0.000000 |
| test | conditional_alpha_raw | 1.605176 | 1.606977 | 0.001801 | 0.998879 | 0.600706 | 0.182508 |
| test | conditional_alpha_mixed_with_global | 1.598818 | 1.606977 | 0.008159 | 0.994923 | 0.374442 | 0.010950 |
| test | conditional_alpha_affine_calibrated | 1.598731 | 1.606977 | 0.008246 | 0.994869 | 0.364099 | 0.025551 |
| test | oracle_per_sample_alpha | 1.581165 | 1.606977 | 0.025812 | 0.983938 | 0.275537 | 0.328127 |

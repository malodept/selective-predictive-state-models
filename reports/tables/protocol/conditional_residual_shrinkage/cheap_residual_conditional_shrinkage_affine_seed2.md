# Conditional residual shrinkage with affine validation calibration

Prediction family:

`z_pred = z_current + alpha(x) * delta_hat`

Global alpha selected on validation: `0.3500`.
Mixture coefficient selected on validation: `gamma=0.1400`.
Affine calibration selected on validation: `alpha_cal = clip(0.1800 + 0.3100 * alpha_raw, 0, 1)`.

| split | method | error | identity error | improvement vs identity | ratio vs identity | alpha mean | alpha std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val | identity | 1.432184 | 1.432184 | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| val | raw_residual_alpha_1 | 1.451261 | 1.432184 | -0.019077 | 1.013320 | 1.000000 | 0.000000 |
| val | global_alpha_val | 1.423970 | 1.432184 | 0.008215 | 0.994264 | 0.350000 | 0.000000 |
| val | conditional_alpha_raw | 1.427803 | 1.432184 | 0.004381 | 0.996941 | 0.552929 | 0.185303 |
| val | conditional_alpha_mixed_with_global | 1.423858 | 1.432184 | 0.008326 | 0.994186 | 0.378410 | 0.025942 |
| val | conditional_alpha_affine_calibrated | 1.423751 | 1.432184 | 0.008434 | 0.994111 | 0.351408 | 0.057444 |
| val | oracle_per_sample_alpha | 1.403749 | 1.432184 | 0.028435 | 0.980145 | 0.274731 | 0.346442 |
| test | identity | 1.606977 | 1.606977 | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| test | raw_residual_alpha_1 | 1.620466 | 1.606977 | -0.013489 | 1.008394 | 1.000000 | 0.000000 |
| test | global_alpha_val | 1.599173 | 1.606977 | 0.007804 | 0.995144 | 0.350000 | 0.000000 |
| test | conditional_alpha_raw | 1.601292 | 1.606977 | 0.005685 | 0.996462 | 0.536137 | 0.174360 |
| test | conditional_alpha_mixed_with_global | 1.599000 | 1.606977 | 0.007976 | 0.995036 | 0.376059 | 0.024410 |
| test | conditional_alpha_affine_calibrated | 1.598973 | 1.606977 | 0.008004 | 0.995019 | 0.346203 | 0.054052 |
| test | oracle_per_sample_alpha | 1.583670 | 1.606977 | 0.023307 | 0.985496 | 0.292246 | 0.335154 |

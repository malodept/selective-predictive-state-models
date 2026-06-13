# Conditional residual shrinkage with affine validation calibration

Prediction family:

`z_pred = z_current + alpha(x) * delta_hat`

Global alpha selected on validation: `0.3800`.
Mixture coefficient selected on validation: `gamma=0.0900`.
Affine calibration selected on validation: `alpha_cal = clip(0.2400 + 0.2300 * alpha_raw, 0, 1)`.

| split | method | error | identity error | improvement vs identity | ratio vs identity | alpha mean | alpha std |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val | identity | 1.432184 | 1.432184 | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| val | raw_residual_alpha_1 | 1.448158 | 1.432184 | -0.015973 | 1.011153 | 1.000000 | 0.000000 |
| val | global_alpha_val | 1.423151 | 1.432184 | 0.009034 | 0.993692 | 0.380000 | 0.000000 |
| val | conditional_alpha_raw | 1.427603 | 1.432184 | 0.004581 | 0.996801 | 0.585426 | 0.188142 |
| val | conditional_alpha_mixed_with_global | 1.423107 | 1.432184 | 0.009077 | 0.993662 | 0.398488 | 0.016933 |
| val | conditional_alpha_affine_calibrated | 1.423023 | 1.432184 | 0.009161 | 0.993603 | 0.374648 | 0.043273 |
| val | oracle_per_sample_alpha | 1.402067 | 1.432184 | 0.030118 | 0.978971 | 0.282141 | 0.348442 |
| test | identity | 1.606977 | 1.606977 | 0.000000 | 1.000000 | 0.000000 | 0.000000 |
| test | raw_residual_alpha_1 | 1.626890 | 1.606977 | -0.019913 | 1.012392 | 1.000000 | 0.000000 |
| test | global_alpha_val | 1.599954 | 1.606977 | 0.007022 | 0.995630 | 0.380000 | 0.000000 |
| test | conditional_alpha_raw | 1.604770 | 1.606977 | 0.002207 | 0.998627 | 0.589385 | 0.175677 |
| test | conditional_alpha_mixed_with_global | 1.599991 | 1.606977 | 0.006986 | 0.995653 | 0.398845 | 0.015811 |
| test | conditional_alpha_affine_calibrated | 1.599790 | 1.606977 | 0.007187 | 0.995528 | 0.375559 | 0.040406 |
| test | oracle_per_sample_alpha | 1.584239 | 1.606977 | 0.022738 | 0.985851 | 0.274036 | 0.326761 |

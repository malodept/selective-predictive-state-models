# Ensemble uncertainty shrinkage diagnostic

This diagnostic uses the disagreement of independently trained residual predictors as a deployment-time uncertainty signal for residual shrinkage.

| split | method | error | identity error | improvement vs identity | ratio vs identity | alpha mean | alpha std | alpha Spearman oracle | score | n bins | gamma |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| val | identity | 1.432184 | 1.432184 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| val | ensemble_raw_alpha_1 | 1.443616 | 1.432184 | -0.011432 | 1.007982 | 1.000000 | 0.000000 | nan | nan | nan | nan |
| val | ensemble_global_alpha_val | 1.422280 | 1.432184 | 0.009904 | 0.993085 | 0.405232 | 0.000000 | nan | nan | nan | nan |
| val | ensemble_uncertainty_calibrated | 1.421332 | 1.432184 | 0.010853 | 0.992422 | 0.433946 | 0.160951 | 0.14450652338606734 | disagreement | 8.0 | 0.9900 |
| val | ensemble_oracle_per_sample_alpha | 1.400897 | 1.432184 | 0.031287 | 0.978154 | 0.296092 | 0.357495 | 1.0 | nan | nan | nan |
| test | identity | 1.606977 | 1.606977 | 0.000000 | 1.000000 | 0.000000 | 0.000000 | nan | nan | nan | nan |
| test | ensemble_raw_alpha_1 | 1.618250 | 1.606977 | -0.011273 | 1.007015 | 1.000000 | 0.000000 | nan | nan | nan | nan |
| test | ensemble_global_alpha_val | 1.598285 | 1.606977 | 0.008692 | 0.994591 | 0.405232 | 0.000000 | nan | nan | nan | nan |
| test | ensemble_uncertainty_calibrated | 1.599601 | 1.606977 | 0.007376 | 0.995410 | 0.433946 | 0.160951 | -0.02700126254941435 | disagreement | 8.0 | 0.9900 |
| test | ensemble_oracle_per_sample_alpha | 1.582063 | 1.606977 | 0.024914 | 0.984497 | 0.297254 | 0.337324 | 1.0 | nan | nan | nan |

# Conditional alpha ranking diagnostics

This diagnostic measures whether the conditional alpha model ranks local oracle shrinkage coefficients.

| split | metric | mean ± std over seeds |
| --- | --- | ---: |
| val | alpha_raw_mean | 0.576057 ± 0.020150 |
| val | alpha_affine_mean | 0.362877 ± 0.011623 |
| val | alpha_oracle_mean | 0.279759 ± 0.004356 |
| val | pearson_raw_oracle | 0.119426 ± 0.020191 |
| val | spearman_raw_oracle | 0.112452 ± 0.010888 |
| val | auc_top25_oracle_alpha_raw | 0.581465 ± 0.012103 |
| val | identity_error | 1.432184 ± 0.000000 |
| val | global_error | 1.423276 ± 0.000640 |
| val | affine_cond_error | 1.423143 ± 0.000558 |
| val | oracle_error | 1.401878 ± 0.001973 |
| val | global_improvement_vs_identity | 0.008908 ± 0.000640 |
| val | affine_improvement_vs_identity | 0.009042 ± 0.000558 |
| val | affine_improvement_vs_global | 0.000133 ± 0.000083 |
| val | oracle_improvement_vs_global | 0.021398 ± 0.001363 |
| test | alpha_raw_mean | 0.575409 ± 0.034478 |
| test | alpha_affine_mean | 0.361953 ± 0.014795 |
| test | alpha_oracle_mean | 0.280606 ± 0.010108 |
| test | pearson_raw_oracle | 0.146409 ± 0.011659 |
| test | spearman_raw_oracle | 0.163907 ± 0.010343 |
| test | auc_top25_oracle_alpha_raw | 0.601723 ± 0.004773 |
| test | identity_error | 1.606977 ± 0.000000 |
| test | global_error | 1.599317 ± 0.000579 |
| test | affine_cond_error | 1.599164 ± 0.000555 |
| test | oracle_error | 1.583025 ± 0.001635 |
| test | global_improvement_vs_identity | 0.007659 ± 0.000579 |
| test | affine_improvement_vs_identity | 0.007812 ± 0.000555 |
| test | affine_improvement_vs_global | 0.000153 ± 0.000054 |
| test | oracle_improvement_vs_global | 0.016293 ± 0.001189 |

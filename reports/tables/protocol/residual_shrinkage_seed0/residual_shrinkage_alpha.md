# Residual shrinkage alpha diagnostic

Prediction family: `z_pred(alpha) = z_current + alpha * delta_hat`, with `alpha in [0, 1]`.

| split | model | rule | alpha | error | identity error | improvement vs identity | ratio vs identity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| val | cheap_residual | raw_alpha_1 | 1.0000 | 1.451741 | 1.432185 | -0.019557 | 1.013655 |
| val | cheap_residual | oracle_best_alpha_on_this_split | 0.3600 | 1.422708 | 1.432185 | 0.009477 | 0.993383 |
| val | cheap_residual | alpha_selected_on_val | 0.3600 | 1.422708 | 1.432185 | 0.009477 | 0.993383 |
| val | expensive_residual | raw_alpha_1 | 1.0000 | 1.443634 | 1.432185 | -0.011450 | 1.007995 |
| val | expensive_residual | oracle_best_alpha_on_this_split | 0.3400 | 1.428077 | 1.432185 | 0.004108 | 0.997132 |
| val | expensive_residual | alpha_selected_on_val | 0.3400 | 1.428077 | 1.432185 | 0.004108 | 0.997132 |
| test | cheap_residual | raw_alpha_1 | 1.0000 | 1.628799 | 1.606977 | -0.021822 | 1.013580 |
| test | cheap_residual | oracle_best_alpha_on_this_split | 0.3400 | 1.598805 | 1.606977 | 0.008172 | 0.994915 |
| test | cheap_residual | alpha_selected_on_val | 0.3600 | 1.598825 | 1.606977 | 0.008152 | 0.994927 |
| test | expensive_residual | raw_alpha_1 | 1.0000 | 1.619689 | 1.606977 | -0.012712 | 1.007910 |
| test | expensive_residual | oracle_best_alpha_on_this_split | 0.3000 | 1.604169 | 1.606977 | 0.002807 | 0.998253 |
| test | expensive_residual | alpha_selected_on_val | 0.3400 | 1.604224 | 1.606977 | 0.002753 | 0.998287 |

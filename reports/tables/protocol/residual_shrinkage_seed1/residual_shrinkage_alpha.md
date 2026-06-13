# Residual shrinkage alpha diagnostic

Prediction family: `z_pred(alpha) = z_current + alpha * delta_hat`, with `alpha in [0, 1]`.

| split | model | rule | alpha | error | identity error | improvement vs identity | ratio vs identity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| val | cheap_residual | raw_alpha_1 | 1.0000 | 1.448158 | 1.432185 | -0.015973 | 1.011153 |
| val | cheap_residual | oracle_best_alpha_on_this_split | 0.3800 | 1.423151 | 1.432185 | 0.009034 | 0.993692 |
| val | cheap_residual | alpha_selected_on_val | 0.3800 | 1.423151 | 1.432185 | 0.009034 | 0.993692 |
| val | expensive_residual | raw_alpha_1 | 1.0000 | 1.448026 | 1.432185 | -0.015842 | 1.011061 |
| val | expensive_residual | oracle_best_alpha_on_this_split | 0.3300 | 1.427094 | 1.432185 | 0.005091 | 0.996445 |
| val | expensive_residual | alpha_selected_on_val | 0.3300 | 1.427094 | 1.432185 | 0.005091 | 0.996445 |
| test | cheap_residual | raw_alpha_1 | 1.0000 | 1.626890 | 1.606977 | -0.019913 | 1.012392 |
| test | cheap_residual | oracle_best_alpha_on_this_split | 0.3400 | 1.599851 | 1.606977 | 0.007125 | 0.995566 |
| test | cheap_residual | alpha_selected_on_val | 0.3800 | 1.599954 | 1.606977 | 0.007022 | 0.995630 |
| test | expensive_residual | raw_alpha_1 | 1.0000 | 1.621305 | 1.606977 | -0.014328 | 1.008916 |
| test | expensive_residual | oracle_best_alpha_on_this_split | 0.3100 | 1.603312 | 1.606977 | 0.003665 | 0.997720 |
| test | expensive_residual | alpha_selected_on_val | 0.3300 | 1.603326 | 1.606977 | 0.003651 | 0.997728 |

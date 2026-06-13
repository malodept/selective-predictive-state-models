# Residual shrinkage alpha diagnostic

Prediction family: `z_pred(alpha) = z_current + alpha * delta_hat`, with `alpha in [0, 1]`.

| split | model | rule | alpha | error | identity error | improvement vs identity | ratio vs identity |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| val | cheap_residual | raw_alpha_1 | 1.0000 | 1.451261 | 1.432185 | -0.019077 | 1.013320 |
| val | cheap_residual | oracle_best_alpha_on_this_split | 0.3500 | 1.423970 | 1.432185 | 0.008215 | 0.994264 |
| val | cheap_residual | alpha_selected_on_val | 0.3500 | 1.423970 | 1.432185 | 0.008215 | 0.994264 |
| val | expensive_residual | raw_alpha_1 | 1.0000 | 1.450238 | 1.432185 | -0.018053 | 1.012605 |
| val | expensive_residual | oracle_best_alpha_on_this_split | 0.3100 | 1.427864 | 1.432185 | 0.004320 | 0.996983 |
| val | expensive_residual | alpha_selected_on_val | 0.3100 | 1.427864 | 1.432185 | 0.004320 | 0.996983 |
| test | cheap_residual | raw_alpha_1 | 1.0000 | 1.620466 | 1.606977 | -0.013490 | 1.008394 |
| test | cheap_residual | oracle_best_alpha_on_this_split | 0.3800 | 1.599132 | 1.606977 | 0.007845 | 0.995118 |
| test | cheap_residual | alpha_selected_on_val | 0.3500 | 1.599173 | 1.606977 | 0.007803 | 0.995144 |
| test | expensive_residual | raw_alpha_1 | 1.0000 | 1.621353 | 1.606977 | -0.014376 | 1.008946 |
| test | expensive_residual | oracle_best_alpha_on_this_split | 0.3100 | 1.603336 | 1.606977 | 0.003641 | 0.997734 |
| test | expensive_residual | alpha_selected_on_val | 0.3100 | 1.603336 | 1.606977 | 0.003641 | 0.997734 |

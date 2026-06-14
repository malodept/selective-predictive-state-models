# Directional residual predictor summary

Loss: `MSE + 0.05 * cosine_loss + 0.001 * lognorm_loss`.

| split | seeds | identity error | raw error | global alpha | global error | improvement vs identity | cosine mean | cosine positive frac | true Δ norm med | pred Δ norm med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | 3 | 1.432185 ± 0.000000 | 1.836174 ± 0.055106 | 0.172075 ± 0.012606 | 1.414079 ± 0.001336 | 0.018106 ± 0.001336 | 0.090013 ± 0.003637 | 0.828405 ± 0.007795 | 14.341398 ± 0.000000 | 14.994472 ± 0.863892 |
| test | 3 | 1.606977 ± 0.000000 | 2.006061 ± 0.048345 | 0.168880 ± 0.010167 | 1.589900 ± 0.000645 | 0.017077 ± 0.000645 | 0.086728 ± 0.001894 | 0.854629 ± 0.006473 | 16.493105 ± 0.000000 | 14.589526 ± 0.653513 |

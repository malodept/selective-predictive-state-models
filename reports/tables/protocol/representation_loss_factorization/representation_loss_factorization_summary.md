# Representation/loss factorization summary

This table isolates two factors: the latent representation (`CLS` vs `[CLS, patch_mean, patch_std]`) and the residual training loss (`MSE` vs directional).

| representation | loss | identity error | calibrated gain | relative gain | cosine mean | positive cosine frac |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| CLS | MSE residual | 1.606977 | 0.007659 ± 0.000579 | 0.477% | 0.057030 | 0.757434 |
| CLS | directional residual | 1.606977 | 0.017654 ± 0.000224 | 1.099% | 0.088062 | 0.857781 |
| spatial | MSE residual | 0.727133 | 0.007356 ± 0.000311 | 1.012% | 0.087571 | 0.886191 |
| spatial | directional residual | 0.727133 | 0.010625 ± 0.000179 | 1.461% | 0.105416 | 0.928168 |

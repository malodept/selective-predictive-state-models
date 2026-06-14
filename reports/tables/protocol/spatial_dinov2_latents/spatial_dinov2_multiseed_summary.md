# Spatial DINOv2 latent multiseed summary

This table evaluates DINOv2 spatial latents `[CLS, patch_mean, patch_std]` under the same environment-held-out protocol.

| method | seeds | identity error | raw error | global alpha | global error | improvement vs identity | relative improvement | cosine mean | positive cosine frac | true Δ norm med | pred Δ norm med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| spatial MSE-only | 3 | 0.727133 ± 0.000000 | 0.916771 ± 0.025030 | 0.162420 ± 0.010125 | 0.719777 ± 0.000311 | 0.007356 ± 0.000311 | 1.011591 ± 0.042760% | 0.087571 ± 0.001450 | 0.886191 ± 0.005351 | 21.756792 ± 0.000000 | 17.619642 ± 0.890039 |
| spatial directional λcos=0.01 | 3 | 0.727133 ± 0.000000 | 1.051733 ± 0.033415 | 0.151399 ± 0.005602 | 0.716508 ± 0.000179 | 0.010625 ± 0.000179 | 1.461195 ± 0.024615% | 0.105416 ± 0.001498 | 0.928168 ± 0.003289 | 21.756792 ± 0.000000 | 22.007718 ± 0.975662 |

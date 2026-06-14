# Spatial DINOv2 latent diagnostic

This diagnostic compares the original DINOv2 CLS latent with a spatial DINOv2 latent built as `[CLS, patch_mean, patch_std]`.

| method | identity error | raw error | global alpha | global error | improvement vs identity | relative improvement | cosine mean | positive cosine frac | true Δ norm med | pred Δ norm med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLS directional λcos=0.01 | 1.606977 | 2.103272 | 0.155716 | 1.589500 | 0.017477 | 1.088% | 0.087518 | 0.851452 | 16.493105 | 15.848438 |
| spatial MSE-only | 0.727133 | 0.904645 | 0.164652 | 0.719957 | 0.007175 | 0.987% | 0.086931 | 0.884680 | 21.756792 | 17.189388 |
| spatial directional λcos=0.01 | 0.727133 | 1.013597 | 0.157774 | 0.716714 | 0.010418 | 1.433% | 0.103728 | 0.927324 | 21.756792 | 20.888390 |

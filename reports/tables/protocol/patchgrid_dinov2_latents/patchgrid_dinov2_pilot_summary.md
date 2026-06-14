# Patch-grid DINOv2 latent pilot

This pilot compares the previous CLS and spatial-statistics DINOv2 latents against a projected patch-grid latent.

Patch-grid latent: `[CLS, patch_mean, patch_std, random_projection(pool_4x4(patch_tokens))]`.

| method | identity error | raw error | global alpha | global error | gain vs identity | relative gain | cosine mean | positive cosine frac | true Δ norm med | pred Δ norm med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CLS directional λcos=0.01 | 1.606977 | 2.103272 | 0.155716 | 1.589500 | 0.017477 | 1.088% | 0.087518 | 0.851452 | 16.493105 | 15.848438 |
| spatial stats MSE-only | 0.727133 | 0.904645 | 0.164652 | 0.719957 | 0.007175 | 0.987% | 0.086931 | 0.884680 | 21.756792 | 17.189388 |
| spatial stats directional λcos=0.01 | 0.727133 | 1.013597 | 0.157774 | 0.716714 | 0.010418 | 1.433% | 0.103728 | 0.927324 | 21.756792 | 20.888390 |
| patch-grid MSE-only | 3.517845 | 4.103095 | 0.139017 | 3.502178 | 0.015666 | 0.445% | 0.055899 | 0.833268 | 63.504494 | 41.484234 |
| patch-grid directional λcos=0.01 | 3.517845 | 4.669168 | 0.143846 | 3.484400 | 0.033444 | 0.951% | 0.082559 | 0.907807 | 63.504494 | 56.357445 |

## Interpretation

- The patch-grid representation improves over identity after calibration.
- Directional loss strongly improves patch-grid dynamics compared with MSE-only.
- However, projected patch-grid latents do not outperform spatial-statistics latents in relative calibrated gain.
- This suggests that preserving patch-level information is not sufficient if the downstream predictor collapses it into a global vector; a true patch-token or attention-based dynamics model is likely required.

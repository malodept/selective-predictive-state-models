# Patch-token Action Transformer pilot

This pilot compares global-vector DINOv2 dynamics with a token-preserving action-conditioned transformer.

| method | identity error | raw error | global alpha | global error | gain vs identity | relative gain | cosine mean | positive cosine frac | true Δ norm med | pred Δ norm med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patch-token transformer MSE-only | 1.109343 | 1.097957 | 0.625000 | 1.062886 | 0.046457 | 4.188% | 0.183513 | 0.997254 | 59.583115 | 31.862093 |
| patch-token transformer directional | 1.109343 | 1.097805 | 0.625000 | 1.064012 | 0.045331 | 4.086% | 0.181086 | 0.997195 | 59.583115 | 31.371330 |
| patch-grid vector MSE-only | 3.517845 | 4.103095 | 0.139017 | 3.502178 | 0.015666 | 0.445% | 0.055899 | 0.833268 | 63.504494 | 41.484234 |
| patch-grid vector directional | 3.517845 | 4.669168 | 0.143846 | 3.484400 | 0.033444 | 0.951% | 0.082559 | 0.907807 | 63.504494 | 56.357445 |
| spatial stats directional | 0.727133 | 1.013597 | 0.157774 | 0.716714 | 0.010418 | 1.433% | 0.103728 | 0.927324 | 21.756792 | 20.888390 |
| CLS directional | 1.606977 | 2.103272 | 0.155716 | 1.589500 | 0.017477 | 1.088% | 0.087518 | 0.851452 | 16.493105 | 15.848438 |

## Interpretation

- The patch-token transformer is substantially stronger than vectorized patch-grid dynamics.
- MSE-only slightly outperforms the directional variant in this pilot.
- The main effect therefore comes from preserving token structure and using attention, not from the auxiliary directional loss.
- The high positive cosine fraction suggests that the transformer predicts latent displacements in the correct half-space for nearly all test samples.

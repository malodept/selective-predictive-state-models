# Patch-token attention ablation

Both models receive the same 16 DINOv2 patch tokens and the same 7D action. The MLP predicts each token independently, while the Transformer allows token-token attention.

| model | seeds | identity error | global error | gain vs identity | relative gain | cosine mean | positive cosine frac | global alpha | pred Δ norm med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patch-token MLP, no token attention | 3 | 1.109343 ± 0.000000 | 1.088086 ± 0.001503 | 0.021256 ± 0.001503 | 1.916133 ± 0.135467% | 0.127054 ± 0.002407 | 0.983621 ± 0.002263 | 0.538333 ± 0.007638 | 28.217871 ± 0.643453 |
| patch-token Transformer, token attention | 3 | 1.109343 ± 0.000000 | 1.065375 ± 0.002237 | 0.043968 ± 0.002237 | 3.963389 ± 0.201617% | 0.181000 ± 0.002181 | 0.997293 ± 0.000039 | 0.630000 ± 0.008660 | 32.861644 ± 0.955390 |

## Interpretation

- Mean Transformer gain is `0.043968`, compared with `0.021256` for the token-wise MLP.
- The Transformer achieves approximately `2.07x` the calibrated gain of the no-attention token-wise baseline.
- This supports the conclusion that the improvement is not merely due to exposing more DINOv2 patch information; token-token interaction is important for OOD latent dynamics.

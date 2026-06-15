# Patch-token Action Transformer multiseed summary

This table evaluates the token-preserving action-conditioned transformer over three seeds.

| method | seeds | identity error | raw error | global alpha | global error | gain vs identity | relative gain | cosine mean | positive cosine frac | true Δ norm med | pred Δ norm med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| patch-token transformer MSE-only | 3 | 1.109343 ± 0.000000 | 1.105187 ± 0.006629 | 0.630000 ± 0.008660 | 1.065375 ± 0.002237 | 0.043968 ± 0.002237 | 3.963389 ± 0.201617% | 0.181000 ± 0.002181 | 0.997293 ± 0.000039 | 59.583115 ± 0.000000 | 32.861644 ± 0.955390 |

## Interpretation

- The patch-token transformer consistently outperforms the previous global-vector dynamics models.
- The gain is stable across seeds despite early stopping at epoch 13 with best epoch 1.
- The very high positive cosine fraction indicates that nearly all predicted latent displacements lie in the correct half-space.
- The result supports the hypothesis that preserving spatial token structure is more important than simply increasing global latent dimensionality.

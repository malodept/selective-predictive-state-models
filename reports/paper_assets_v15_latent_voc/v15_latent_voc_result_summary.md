# SPSM v15 latent-state-aware value-of-computation result

This result updates the SPSM value-of-computation story using locked router families rather than per-cell feature selection.

## Main conclusion

Observable latent-state features improve value-of-computation routing on the hardest long-horizon OOD shift, especially at moderate compute costs.

The strongest case is `3-block, H=72`:

| lambda | cheap | full | confidence | context-only best | context+latent best | oracle |
|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 0.731 | 0.768 | 0.770 | 0.788 | 0.798 | 0.888 |
| 0.10 | 0.731 | 0.718 | 0.756 | 0.763 | 0.781 | 0.879 |
| 0.20 | 0.731 | 0.618 | 0.731 | 0.744 | 0.746 | 0.863 |
| 0.30 | 0.731 | 0.518 | 0.731 | 0.728 | 0.737 | 0.846 |

## Interpretation

The full model is not uniformly better than the cheap model. On `3-block, H=36`, cheap-only remains optimal among fixed non-oracle policies. On `3-block, H=72`, the expensive model can rescue the cheap model often enough to justify selective routing.

The best routing signal is not candidate-mined geometry. The useful signal comes from cheap-model risk, environment context, and observable current-state latent statistics computed from the frozen encoder representation before routing.

This supports framing the method as latent-state-aware value-of-computation rather than simply confidence-based selective prediction.

## Paper update

The paper should replace the previous context-aware routing result with a locked comparison centered on:

- confidence threshold
- context classifier
- KNN expected-gain context router
- rescue-harm context router
- KNN expected-gain context+latent router
- rescue-harm context+latent router

The central table should be `table_v15_h72_locked_router.tex`, with `fig_v15_locked_utility_h72.png` as the main routing figure.

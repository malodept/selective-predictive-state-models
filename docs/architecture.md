# Architecture

The first implementation has four trainable pieces:

1. **Encoder wrapper**: identity in the synthetic setup, later a frozen teacher or compact student encoder.
2. **Future predictor**: maps current latent state plus optional action/context to a future latent state.
3. **Reliability head**: predicts whether the transition is unreliable, corrupted, or unlikely to be useful.
4. **Selector policy**: chooses between cheap-only execution and optional extra compute.

```text
z_t + a_t -> predictor -> z_hat_t+delta
z_t + z_hat_t+delta -> reliability head -> failure probability
z_t + reliability + budget -> selector -> cheap-only or optional compute
```

The v0 selector can be threshold-based. Later versions can make the selector differentiable or utility-trained.

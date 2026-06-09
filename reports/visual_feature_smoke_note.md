# Visual feature smoke test

This experiment is the bridge between the synthetic latent proof of concept and real visual datasets.
It replaces synthetic latent states with frozen features extracted from image sequences, while keeping the same SPSM pipeline:

1. observation image `o_t` -> frozen feature `z_t`;
2. future image `o_{t+Δ}` -> frozen target feature `z_{t+Δ}`;
3. transition context/action `a_t = Δ / max_Δ`;
4. predictor learns `z_t, a_t -> z_hat_{t+Δ}`;
5. reliability predicts whether the transition is expected to be difficult;
6. selector decides whether optional refinement compute is worth paying for.

The default extractor is a dependency-light patch-mean encoder, not a final research encoder. It exists to validate the data flow and evaluation protocol before plugging in DINO/V-JEPA/TartanAir features.

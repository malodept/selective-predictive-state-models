# v48 predicted-latent calibration claim

## Main idea

v48 separates three notions that are often conflated in latent world models:
1. counterfactual latent discrimination;
2. physical calibration of predicted latents;
3. downstream decision actionability.

## Scientific claim

Predicted latent futures can contain physical information without being directly decodable by probes trained on true encoded futures. In SPSM, a true-future xy probe decodes true future latents accurately, but fails on predicted future latents. Model-specific calibration recovers accurate physical xy predictions and improves one-step action selection.

## Why this matters

This means that latent ranking correctness is not enough to guarantee actionability. A predicted latent can be useful for discriminating the correct counterfactual future while still living in a representation coordinate system that requires calibration before downstream planning.

## How it strengthens the paper

v47 shows that predictive compute has query-level regimes. v48 shows that predicted latent usefulness also has calibration regimes. Together, they support the broader thesis: predictive usefulness is multi-dimensional, not a scalar accuracy.

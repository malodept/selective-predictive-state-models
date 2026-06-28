# v30 recommended paper framing

Main claim:
SPSM is a controlled exact-intervention protocol for evaluating reliability and value-of-computation in action-conditioned latent world models.

Empirical claims supported by current results:
1. State-action latent dynamics are learned under state-disjoint exact interventions.
2. OOD degradation is driven more by constrained geometry than by horizon or velocity alone.
3. Cheap and expensive predictors are not uniformly ordered, making VoC non-trivial.
4. Capacity is non-monotonic under hard OOD: Medium can beat Full.
5. Measured latency changes the compute frontier relative to checkpoint-size proxies.
6. Learned multi-capacity routing gives stable gains on H=48, but not on H=36 or H=72.
7. H=72 remains an oracle-headroom problem, not a solved routing problem.

Claims to avoid:
- Do not claim general adaptive computation is solved.
- Do not claim larger models are generally worse.
- Do not claim latent features always help.
- Do not claim the router is robust across all OOD regimes.

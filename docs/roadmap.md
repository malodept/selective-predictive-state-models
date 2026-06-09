# Roadmap

## Week 1

- Synthetic latent dataset.
- MLP future predictor.
- Reliability head.
- Retrieval and surprise metrics.
- First utility-vs-compute table.

## Week 2

- Public trajectory loader skeleton.
- Frozen feature extraction interface.
- Adaptive selector ablations.
- First figure-ready plot.

## Weeks 3-4

- Teacher-feature baseline on public data.
- Multi-step prediction.
- Reliability calibration.
- Latency and model-size sweep.

## Weeks 5-8

- Real transfer data.
- Stronger baselines.
- Technical report.
- Outreach package.

## v0.5 visual feature smoke test

The next milestone after the synthetic proof of concept is a minimal visual-feature pipeline:

```text
image sequence -> frozen feature extractor -> z_t, z_t+Δ -> SPSM predictor/reliability/selector
```

The repository includes a dependency-light sample image generator and patch-mean encoder so the full pipeline can be run without external datasets. This is not the final encoder; it is a smoke test before using stronger frozen encoders and public datasets.

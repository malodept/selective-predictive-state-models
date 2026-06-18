# Toy exact-intervention dataset audit

- output: `outputs/counterfactual/toy_exact/toy_exact_intervention_v0.npz`
- groups: `1000`
- candidates per group: `5`
- samples: `5000`
- chance top-1: `0.200000`
- latent shape: `(16, 8)`
- action dimension: `2`
- image size: `64`
- step: `8`

## Interpretation

Each group is an exact-intervention branch: the same synthetic state is reset and branched under five different actions.
This is not meant to be a realistic benchmark. It is a protocol sanity check: the oracle should be far above chance, ideally exactly 1.0 at alpha=1.

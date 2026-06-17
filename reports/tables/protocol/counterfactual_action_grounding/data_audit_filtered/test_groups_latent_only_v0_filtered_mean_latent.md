# Filtered matched-state group audit

- input: `outputs/counterfactual/tartanair_silver/test_groups_latent_only_v0.npz`
- score: `mean_latent`
- total groups: `8000`

| keep frac | groups | score max | latent mean | action mean | future mean | same traj offdiag | same env offdiag |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 400 | 0.563463 | 0.508542 | 2.600079 | 24.694805 | 0.000000 | 1.000000 |
| 0.10 | 800 | 0.608170 | 0.546853 | 2.472999 | 23.920416 | 0.000000 | 1.000000 |
| 0.20 | 1600 | 0.665728 | 0.593240 | 2.555643 | 23.589785 | 0.000000 | 1.000000 |

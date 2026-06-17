# Filtered matched-state group audit

- input: `outputs/counterfactual/tartanair_silver/val_groups_latent_only_v0.npz`
- score: `mean_latent`
- total groups: `8000`

| keep frac | groups | score max | latent mean | action mean | future mean | same traj offdiag | same env offdiag |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 400 | 0.646447 | 0.607692 | 4.269121 | 21.859375 | 0.000000 | 1.000000 |
| 0.10 | 800 | 0.687074 | 0.637266 | 3.793964 | 23.802176 | 0.000000 | 1.000000 |
| 0.20 | 1600 | 0.750247 | 0.678622 | 3.559789 | 24.792612 | 0.000000 | 1.000000 |

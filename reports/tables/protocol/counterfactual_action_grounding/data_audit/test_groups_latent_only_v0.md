# TartanAir silver matched-state group audit

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_test.npz`
- output: `outputs/counterfactual/tartanair_silver/test_groups_latent_only_v0.npz`
- mining version: `tartanair_silver_latent_only_v0`
- samples: `50980`
- groups: `8000`
- candidates per group: `5`
- chance top-1: `0.200000`
- pool size: `30000`
- state projection dim: `64`
- nearest pool: `256`
- min action distance: `0.25`
- min future distance: `0.02`

## Quality-control summary

| quantity | value |
| --- | ---: |
| same trajectory fraction offdiag | 0.000000 |
| same environment fraction offdiag | 1.000000 |
| unique anchor trajectories | 18 |
| unique anchor environments | 1 |

## Off-diagonal candidate distributions

| metric | mean | p05 | p25 | median | p75 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| latent distance | 0.748204 | 0.563463 | 0.684836 | 0.754481 | 0.816219 | 0.919893 |
| action distance | 2.839307 | 0.676660 | 1.472193 | 2.339958 | 3.471919 | 5.947580 |
| future distance | 25.352623 | 19.027177 | 22.128555 | 24.618436 | 27.580977 | 33.986691 |

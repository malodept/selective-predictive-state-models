# TartanAir silver matched-state group audit

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_train.npz`
- output: `outputs/counterfactual/tartanair_silver/train_groups_latent_only_v0.npz`
- mining version: `tartanair_silver_latent_only_v0`
- samples: `74610`
- groups: `20000`
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
| same environment fraction offdiag | 0.954937 |
| unique anchor trajectories | 31 |
| unique anchor environments | 4 |

## Off-diagonal candidate distributions

| metric | mean | p05 | p25 | median | p75 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| latent distance | 0.733336 | 0.534530 | 0.653096 | 0.739131 | 0.820563 | 0.913100 |
| action distance | 2.574074 | 0.509680 | 1.121319 | 1.934840 | 3.197285 | 6.209685 |
| future distance | 24.551695 | 16.883852 | 20.702834 | 24.057985 | 27.610641 | 34.362113 |

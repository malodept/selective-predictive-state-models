# TartanAir silver matched-state group audit

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_val.npz`
- output: `outputs/counterfactual/tartanair_silver/val_groups_latent_only_v0.npz`
- mining version: `tartanair_silver_latent_only_v0`
- samples: `10890`
- groups: `8000`
- candidates per group: `5`
- chance top-1: `0.200000`
- pool size: `10890`
- state projection dim: `64`
- nearest pool: `256`
- min action distance: `0.25`
- min future distance: `0.02`

## Quality-control summary

| quantity | value |
| --- | ---: |
| same trajectory fraction offdiag | 0.000000 |
| same environment fraction offdiag | 1.000000 |
| unique anchor trajectories | 5 |
| unique anchor environments | 1 |

## Off-diagonal candidate distributions

| metric | mean | p05 | p25 | median | p75 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| latent distance | 0.855475 | 0.646447 | 0.777993 | 0.873685 | 0.939442 | 1.022977 |
| action distance | 3.220903 | 0.863811 | 1.806627 | 2.754600 | 3.906745 | 6.670316 |
| future distance | 28.849680 | 19.496897 | 24.080311 | 27.355251 | 32.147373 | 43.299085 |

# Pose audit for matched-state groups

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_val.npz`
- groups: `outputs/counterfactual/tartanair_silver/val_groups_latent_only_v0.npz`
- group count: `8000`
- candidates: `5`

| metric | mean | p05 | p25 | median | p75 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| position distance | 17.859413 | 2.455495 | 10.405484 | 16.940291 | 24.314329 | 37.868870 | 53.500202 |
| velocity distance | 1.136844 | 0.242227 | 0.600160 | 1.007366 | 1.548871 | 2.468164 | 4.109792 |
| quaternion dot distance | 0.333380 | 0.005804 | 0.039889 | 0.203514 | 0.636023 | 0.939333 | 0.998965 |

# Pose audit for matched-state groups

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_train.npz`
- groups: `outputs/counterfactual/tartanair_silver/train_groups_latent_only_v0.npz`
- group count: `20000`
- candidates: `5`

| metric | mean | p05 | p25 | median | p75 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| position distance | 11.981921 | 1.004187 | 4.257642 | 9.404141 | 15.842753 | 32.764744 | 75.440582 |
| velocity distance | 0.713582 | 0.074163 | 0.260781 | 0.556485 | 0.992316 | 1.928254 | 4.708582 |
| quaternion dot distance | 0.318011 | 0.002731 | 0.031723 | 0.205726 | 0.510434 | 0.951010 | 0.999987 |

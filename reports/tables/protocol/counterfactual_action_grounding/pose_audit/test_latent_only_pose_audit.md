# Pose audit for matched-state groups

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_test.npz`
- groups: `outputs/counterfactual/tartanair_silver/test_groups_latent_only_v0.npz`
- group count: `8000`
- candidates: `5`

| metric | mean | p05 | p25 | median | p75 | p95 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| position distance | 80.928433 | 12.146356 | 40.429356 | 75.526161 | 113.060490 | 172.233981 | 224.265152 |
| velocity distance | 1.092006 | 0.154658 | 0.502840 | 0.940008 | 1.546734 | 2.532340 | 4.592833 |
| quaternion dot distance | 0.339333 | 0.007293 | 0.054894 | 0.214391 | 0.603111 | 0.944944 | 0.999870 |

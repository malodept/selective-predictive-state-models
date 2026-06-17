# TartanAir silver pose-aware group audit

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_test.npz`
- output: `outputs/counterfactual/tartanair_silver_poseaware/test_groups_poseaware_v1_strict.npz`
- groups: `168`
- candidates per group: `5`
- chance top-1: `0.200000`
- max position distance: `2.0`
- max velocity distance: `0.5`
- max quaternion distance: `0.05`
- min action distance: `0.25`
- min future distance: `0.02`

## Quality-control summary

| quantity | value |
| --- | ---: |
| same trajectory fraction offdiag | 0.000000 |
| same environment fraction offdiag | 1.000000 |
| unique anchor trajectories | 5 |
| unique anchor environments | 1 |

## Off-diagonal distributions

| metric | mean | p05 | p25 | median | p75 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| position distance | 1.364566 | 0.690694 | 1.111468 | 1.395376 | 1.645604 | 1.907016 |
| velocity distance | 0.324867 | 0.112891 | 0.229851 | 0.332018 | 0.431630 | 0.483624 |
| quaternion distance | 0.021910 | 0.004008 | 0.013644 | 0.019905 | 0.029292 | 0.046565 |
| latent distance | 5.076349 | 3.561543 | 4.432339 | 5.005635 | 5.720814 | 6.584134 |
| action distance | 1.488411 | 0.503157 | 0.797625 | 1.149728 | 1.964867 | 3.465950 |
| future distance | 27.964416 | 18.157459 | 24.649438 | 27.730309 | 32.073696 | 36.667905 |

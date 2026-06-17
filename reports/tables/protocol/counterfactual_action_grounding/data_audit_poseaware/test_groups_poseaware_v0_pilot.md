# TartanAir silver pose-aware group audit

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_test.npz`
- output: `outputs/counterfactual/tartanair_silver_poseaware/test_groups_poseaware_v0_pilot.npz`
- groups: `500`
- candidates per group: `5`
- chance top-1: `0.200000`
- max position distance: `5.0`
- max velocity distance: `1.0`
- max quaternion distance: `0.15`
- min action distance: `0.25`
- min future distance: `0.02`

## Quality-control summary

| quantity | value |
| --- | ---: |
| same trajectory fraction offdiag | 0.000000 |
| same environment fraction offdiag | 1.000000 |
| unique anchor trajectories | 16 |
| unique anchor environments | 1 |

## Off-diagonal distributions

| metric | mean | p05 | p25 | median | p75 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| position distance | 2.763890 | 1.013660 | 1.803716 | 2.717982 | 3.734062 | 4.755952 |
| velocity distance | 0.562012 | 0.156545 | 0.347752 | 0.565429 | 0.777432 | 0.963271 |
| quaternion distance | 0.071865 | 0.012772 | 0.032646 | 0.070579 | 0.107152 | 0.139195 |
| latent distance | 6.383711 | 4.112670 | 5.520135 | 6.336903 | 7.285873 | 8.456982 |
| action distance | 1.979105 | 0.594247 | 1.071494 | 1.503021 | 2.088197 | 3.632200 |
| future distance | 31.529664 | 21.314813 | 27.698615 | 31.918744 | 35.716105 | 40.386837 |

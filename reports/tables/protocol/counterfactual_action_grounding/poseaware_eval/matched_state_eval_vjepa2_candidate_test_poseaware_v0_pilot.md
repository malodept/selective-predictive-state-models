# Matched-state group evaluation: vjepa2_candidate_test_poseaware_v0_pilot

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_test.npz`
- groups: `outputs/counterfactual/tartanair_silver_poseaware/test_groups_poseaware_v0_pilot.npz`
- checkpoint: `outputs/envsplit_vjepa2_1_base384_causal16_candidate_seed0_lmatch02/checkpoint.pt`
- candidate groups: `500`
- candidates: `5`
- chance top-1: `0.200000`

| mode | alpha | top-1 | mean rank | positive margin frac | mean margin | diag MSE | best offdiag MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.109303 | 0.164178 | 0.054875 |
| original | 0.25 | 0.200000 | 2.995200 | 0.200000 | -0.105209 | 0.157634 | 0.052425 |
| original | 0.50 | 0.200000 | 2.994000 | 0.200000 | -0.101118 | 0.152707 | 0.051589 |
| original | 0.75 | 0.200000 | 2.992800 | 0.200000 | -0.097031 | 0.149399 | 0.052368 |
| original | 1.00 | 0.200000 | 2.992400 | 0.200000 | -0.092948 | 0.147708 | 0.054759 |
| zero | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.109303 | 0.164178 | 0.054875 |
| zero | 0.25 | 0.200000 | 3.000000 | 0.200000 | -0.108120 | 0.162100 | 0.053980 |
| zero | 0.50 | 0.200000 | 3.000000 | 0.200000 | -0.106937 | 0.160206 | 0.053269 |
| zero | 0.75 | 0.200000 | 3.000000 | 0.200000 | -0.105755 | 0.158496 | 0.052741 |
| zero | 1.00 | 0.200000 | 3.000000 | 0.200000 | -0.104573 | 0.156970 | 0.052397 |
| within_group_shuffle | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.109303 | 0.164178 | 0.054875 |
| within_group_shuffle | 0.25 | 0.200000 | 2.998800 | 0.200000 | -0.105506 | 0.157809 | 0.052302 |
| within_group_shuffle | 0.50 | 0.200000 | 2.999200 | 0.200000 | -0.101714 | 0.153057 | 0.051343 |
| within_group_shuffle | 0.75 | 0.200000 | 2.999200 | 0.200000 | -0.097928 | 0.149924 | 0.051995 |
| within_group_shuffle | 1.00 | 0.200000 | 2.996000 | 0.200000 | -0.094148 | 0.148408 | 0.054260 |
| within_group_reverse | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.109303 | 0.164178 | 0.054875 |
| within_group_reverse | 0.25 | 0.200000 | 3.000400 | 0.200000 | -0.105680 | 0.157902 | 0.052223 |
| within_group_reverse | 0.50 | 0.200000 | 3.005200 | 0.200000 | -0.102060 | 0.153244 | 0.051184 |
| within_group_reverse | 0.75 | 0.200000 | 3.000800 | 0.200000 | -0.098446 | 0.150204 | 0.051759 |
| within_group_reverse | 1.00 | 0.200000 | 2.998400 | 0.200000 | -0.094839 | 0.148782 | 0.053943 |

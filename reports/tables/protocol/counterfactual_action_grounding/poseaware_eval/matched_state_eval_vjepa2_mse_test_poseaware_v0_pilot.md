# Matched-state group evaluation: vjepa2_mse_test_poseaware_v0_pilot

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_test.npz`
- groups: `outputs/counterfactual/tartanair_silver_poseaware/test_groups_poseaware_v0_pilot.npz`
- checkpoint: `outputs/envsplit_vjepa2_1_base384_causal16_transformer_mse_seed0/checkpoint.pt`
- candidate groups: `500`
- candidates: `5`
- chance top-1: `0.200000`

| mode | alpha | top-1 | mean rank | positive margin frac | mean margin | diag MSE | best offdiag MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.109303 | 0.164178 | 0.054875 |
| original | 0.25 | 0.200000 | 2.998000 | 0.200000 | -0.104962 | 0.156899 | 0.051937 |
| original | 0.50 | 0.200000 | 2.995600 | 0.200000 | -0.100625 | 0.151124 | 0.050499 |
| original | 0.75 | 0.200000 | 2.996400 | 0.200000 | -0.096293 | 0.146852 | 0.050559 |
| original | 1.00 | 0.200000 | 2.994400 | 0.200000 | -0.091967 | 0.144085 | 0.052117 |
| zero | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.109303 | 0.164178 | 0.054875 |
| zero | 0.25 | 0.200000 | 3.000000 | 0.200000 | -0.107558 | 0.160891 | 0.053333 |
| zero | 0.50 | 0.200000 | 3.000000 | 0.200000 | -0.105814 | 0.157923 | 0.052109 |
| zero | 0.75 | 0.200000 | 3.000000 | 0.200000 | -0.104069 | 0.155272 | 0.051203 |
| zero | 1.00 | 0.200000 | 3.000000 | 0.200000 | -0.102326 | 0.152940 | 0.050613 |
| within_group_shuffle | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.109303 | 0.164178 | 0.054875 |
| within_group_shuffle | 0.25 | 0.200000 | 3.000800 | 0.200000 | -0.105212 | 0.157053 | 0.051841 |
| within_group_shuffle | 0.50 | 0.200000 | 2.997200 | 0.200000 | -0.101126 | 0.151432 | 0.050306 |
| within_group_shuffle | 0.75 | 0.200000 | 2.998800 | 0.200000 | -0.097045 | 0.147314 | 0.050269 |
| within_group_shuffle | 1.00 | 0.200000 | 2.998400 | 0.200000 | -0.092971 | 0.144701 | 0.051730 |
| within_group_reverse | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.109303 | 0.164178 | 0.054875 |
| within_group_reverse | 0.25 | 0.200000 | 2.999200 | 0.200000 | -0.105341 | 0.157118 | 0.051777 |
| within_group_reverse | 0.50 | 0.200000 | 3.000000 | 0.200000 | -0.101383 | 0.151562 | 0.050179 |
| within_group_reverse | 0.75 | 0.200000 | 3.002000 | 0.200000 | -0.097430 | 0.147509 | 0.050079 |
| within_group_reverse | 1.00 | 0.200000 | 3.002000 | 0.200000 | -0.093482 | 0.144961 | 0.051479 |

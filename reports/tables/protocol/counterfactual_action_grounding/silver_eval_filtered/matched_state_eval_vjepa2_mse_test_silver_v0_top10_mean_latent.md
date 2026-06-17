# Matched-state group evaluation: vjepa2_mse_test_silver_v0_top10_mean_latent

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_test.npz`
- groups: `outputs/counterfactual/tartanair_silver_filtered/test_groups_latent_only_v0_top10_mean_latent.npz`
- checkpoint: `outputs/envsplit_vjepa2_1_base384_causal16_transformer_mse_seed0/checkpoint.pt`
- candidate groups: `800`
- candidates: `5`
- chance top-1: `0.200000`

| mode | alpha | top-1 | mean rank | positive margin frac | mean margin | diag MSE | best offdiag MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.069657 | 0.120192 | 0.050534 |
| original | 0.25 | 0.200250 | 2.994500 | 0.200250 | -0.065905 | 0.113361 | 0.047456 |
| original | 0.50 | 0.200500 | 2.989000 | 0.200500 | -0.062161 | 0.108413 | 0.046252 |
| original | 0.75 | 0.200250 | 2.983250 | 0.200250 | -0.058426 | 0.105349 | 0.046923 |
| original | 1.00 | 0.200000 | 2.967500 | 0.200000 | -0.054710 | 0.104168 | 0.049458 |
| zero | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.069657 | 0.120192 | 0.050534 |
| zero | 0.25 | 0.200000 | 3.000000 | 0.200000 | -0.068274 | 0.117185 | 0.048910 |
| zero | 0.50 | 0.200000 | 3.000000 | 0.200000 | -0.066892 | 0.114526 | 0.047635 |
| zero | 0.75 | 0.200000 | 3.000000 | 0.200000 | -0.065510 | 0.112217 | 0.046707 |
| zero | 1.00 | 0.200000 | 3.000000 | 0.200000 | -0.064133 | 0.110256 | 0.046123 |
| within_group_shuffle | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.069657 | 0.120192 | 0.050534 |
| within_group_shuffle | 0.25 | 0.200000 | 3.001250 | 0.200000 | -0.066260 | 0.113595 | 0.047335 |
| within_group_shuffle | 0.50 | 0.200000 | 2.998750 | 0.200000 | -0.062871 | 0.108882 | 0.046011 |
| within_group_shuffle | 0.75 | 0.200250 | 3.000250 | 0.200250 | -0.059493 | 0.106053 | 0.046560 |
| within_group_shuffle | 1.00 | 0.200000 | 3.000250 | 0.200000 | -0.056136 | 0.105107 | 0.048971 |
| within_group_reverse | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.069657 | 0.120192 | 0.050534 |
| within_group_reverse | 0.25 | 0.199750 | 3.002750 | 0.199750 | -0.066242 | 0.113595 | 0.047354 |
| within_group_reverse | 0.50 | 0.199750 | 3.001750 | 0.199750 | -0.062835 | 0.108883 | 0.046048 |
| within_group_reverse | 0.75 | 0.199500 | 3.006500 | 0.199500 | -0.059438 | 0.106053 | 0.046615 |
| within_group_reverse | 1.00 | 0.200250 | 3.000000 | 0.200250 | -0.056064 | 0.105107 | 0.049044 |

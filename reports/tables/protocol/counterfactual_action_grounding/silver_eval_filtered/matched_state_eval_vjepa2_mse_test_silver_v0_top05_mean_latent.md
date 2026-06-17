# Matched-state group evaluation: vjepa2_mse_test_silver_v0_top05_mean_latent

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_test.npz`
- groups: `outputs/counterfactual/tartanair_silver_filtered/test_groups_latent_only_v0_top05_mean_latent.npz`
- checkpoint: `outputs/envsplit_vjepa2_1_base384_causal16_transformer_mse_seed0/checkpoint.pt`
- candidate groups: `400`
- candidates: `5`
- chance top-1: `0.200000`

| mode | alpha | top-1 | mean rank | positive margin frac | mean margin | diag MSE | best offdiag MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.066363 | 0.114108 | 0.047745 |
| original | 0.25 | 0.200500 | 2.994500 | 0.200500 | -0.062775 | 0.107665 | 0.044889 |
| original | 0.50 | 0.200500 | 2.989000 | 0.200500 | -0.059195 | 0.102975 | 0.043779 |
| original | 0.75 | 0.200000 | 2.985000 | 0.200000 | -0.055624 | 0.100038 | 0.044414 |
| original | 1.00 | 0.200000 | 2.960000 | 0.200000 | -0.052080 | 0.098854 | 0.046775 |
| zero | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.066363 | 0.114108 | 0.047745 |
| zero | 0.25 | 0.200000 | 3.000000 | 0.200000 | -0.065047 | 0.111273 | 0.046226 |
| zero | 0.50 | 0.200000 | 3.000000 | 0.200000 | -0.063732 | 0.108777 | 0.045045 |
| zero | 0.75 | 0.200000 | 3.000000 | 0.200000 | -0.062417 | 0.106620 | 0.044202 |
| zero | 1.00 | 0.200000 | 3.000000 | 0.200000 | -0.061109 | 0.104801 | 0.043692 |
| within_group_shuffle | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.066363 | 0.114108 | 0.047745 |
| within_group_shuffle | 0.25 | 0.200000 | 3.002000 | 0.200000 | -0.063105 | 0.107877 | 0.044772 |
| within_group_shuffle | 0.50 | 0.200500 | 2.999500 | 0.200500 | -0.059855 | 0.103400 | 0.043545 |
| within_group_shuffle | 0.75 | 0.201000 | 3.001000 | 0.201000 | -0.056613 | 0.100675 | 0.044062 |
| within_group_shuffle | 1.00 | 0.200500 | 3.003500 | 0.200500 | -0.053390 | 0.099704 | 0.046314 |
| within_group_reverse | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.066363 | 0.114108 | 0.047745 |
| within_group_reverse | 0.25 | 0.199500 | 3.002000 | 0.199500 | -0.063033 | 0.107849 | 0.044816 |
| within_group_reverse | 0.50 | 0.199500 | 3.003500 | 0.199500 | -0.059712 | 0.103343 | 0.043631 |
| within_group_reverse | 0.75 | 0.198500 | 3.006000 | 0.198500 | -0.056398 | 0.100591 | 0.044193 |
| within_group_reverse | 1.00 | 0.200500 | 2.996000 | 0.200500 | -0.053100 | 0.099592 | 0.046492 |

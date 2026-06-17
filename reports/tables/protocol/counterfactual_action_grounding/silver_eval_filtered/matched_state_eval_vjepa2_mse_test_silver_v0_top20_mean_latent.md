# Matched-state group evaluation: vjepa2_mse_test_silver_v0_top20_mean_latent

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_test.npz`
- groups: `outputs/counterfactual/tartanair_silver_filtered/test_groups_latent_only_v0_top20_mean_latent.npz`
- checkpoint: `outputs/envsplit_vjepa2_1_base384_causal16_transformer_mse_seed0/checkpoint.pt`
- candidate groups: `1600`
- candidates: `5`
- chance top-1: `0.200000`

| mode | alpha | top-1 | mean rank | positive margin frac | mean margin | diag MSE | best offdiag MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.075112 | 0.128067 | 0.052955 |
| original | 0.25 | 0.200125 | 2.996125 | 0.200125 | -0.070999 | 0.120587 | 0.049588 |
| original | 0.50 | 0.200250 | 2.988000 | 0.200250 | -0.066892 | 0.115141 | 0.048249 |
| original | 0.75 | 0.200250 | 2.980250 | 0.200250 | -0.062796 | 0.111731 | 0.048935 |
| original | 1.00 | 0.200000 | 2.965625 | 0.200000 | -0.058717 | 0.110356 | 0.051639 |
| zero | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.075112 | 0.128067 | 0.052955 |
| zero | 0.25 | 0.200000 | 3.000000 | 0.200000 | -0.073596 | 0.124765 | 0.051169 |
| zero | 0.50 | 0.200000 | 3.000000 | 0.200000 | -0.072081 | 0.121843 | 0.049762 |
| zero | 0.75 | 0.200000 | 3.000000 | 0.200000 | -0.070566 | 0.119301 | 0.048735 |
| zero | 1.00 | 0.200000 | 3.000000 | 0.200000 | -0.069056 | 0.117140 | 0.048085 |
| within_group_shuffle | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.075112 | 0.128067 | 0.052955 |
| within_group_shuffle | 0.25 | 0.200000 | 3.002625 | 0.200000 | -0.071350 | 0.120821 | 0.049471 |
| within_group_shuffle | 0.50 | 0.200000 | 3.002000 | 0.200000 | -0.067596 | 0.115609 | 0.048013 |
| within_group_shuffle | 0.75 | 0.200000 | 3.004375 | 0.200000 | -0.063853 | 0.112433 | 0.048580 |
| within_group_shuffle | 1.00 | 0.199875 | 3.002875 | 0.199875 | -0.060128 | 0.111292 | 0.051163 |
| within_group_reverse | 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.075112 | 0.128067 | 0.052955 |
| within_group_reverse | 0.25 | 0.199875 | 3.000000 | 0.199875 | -0.071379 | 0.120838 | 0.049459 |
| within_group_reverse | 0.50 | 0.199875 | 2.999750 | 0.199875 | -0.067654 | 0.115644 | 0.047990 |
| within_group_reverse | 0.75 | 0.199875 | 3.003125 | 0.199875 | -0.063940 | 0.112484 | 0.048545 |
| within_group_reverse | 1.00 | 0.200125 | 3.002250 | 0.200125 | -0.060245 | 0.111360 | 0.051115 |

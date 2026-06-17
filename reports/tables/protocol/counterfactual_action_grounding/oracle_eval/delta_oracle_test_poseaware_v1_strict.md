# Delta-transfer oracle: test_poseaware_v1_strict

This oracle applies each candidate's true latent displacement to the anchor state:

`prediction_k = anchor_current + alpha * (candidate_future_k - candidate_current_k)`

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_test.npz`
- groups: `outputs/counterfactual/tartanair_silver_poseaware/test_groups_poseaware_v1_strict.npz`
- candidate groups: `168`
- candidates: `5`
- chance top-1: `0.200000`

| alpha | top-1 | mean rank | positive margin frac | mean margin |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.077298 |
| 0.25 | 0.200000 | 2.777381 | 0.200000 | -0.069365 |
| 0.50 | 0.200000 | 2.540476 | 0.200000 | -0.061459 |
| 0.75 | 0.203571 | 2.365476 | 0.203571 | -0.053609 |
| 1.00 | 0.214286 | 2.282143 | 0.213095 | -0.045860 |

# Delta-transfer oracle: test_poseaware_v0_pilot

This oracle applies each candidate's true latent displacement to the anchor state:

`prediction_k = anchor_current + alpha * (candidate_future_k - candidate_current_k)`

- data: `outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_test.npz`
- groups: `outputs/counterfactual/tartanair_silver_poseaware/test_groups_poseaware_v0_pilot.npz`
- candidate groups: `500`
- candidates: `5`
- chance top-1: `0.200000`

| alpha | top-1 | mean rank | positive margin frac | mean margin |
| ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.200000 | 3.000000 | 0.200000 | -0.109303 |
| 0.25 | 0.200800 | 2.794800 | 0.200800 | -0.103182 |
| 0.50 | 0.200800 | 2.658800 | 0.200800 | -0.097080 |
| 0.75 | 0.201200 | 2.588000 | 0.200800 | -0.090995 |
| 1.00 | 0.205600 | 2.545200 | 0.204800 | -0.084954 |

# Mixed hard Transformer pairwise audit: state_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard/mixed_hard_v0_10k_groups.npz`
- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard/delta_transformer/state_only_seed0/checkpoint.pt`
- mode: `state_only`
- test groups: `1000`
- top-1: `0.358000`
- mean rank: `1.983000`

| negative type | pairwise accuracy | mean margin | count |
| --- | ---: | ---: | ---: |
| same_state_diff_action | 0.568000 | 0.007143 | 2000 |
| same_action_diff_state | 0.940500 | 0.065569 | 2000 |

## Interpretation

`same_state_diff_action` tests whether the model uses action information.
`same_action_diff_state` tests whether the model uses state-dependent visual dynamics.

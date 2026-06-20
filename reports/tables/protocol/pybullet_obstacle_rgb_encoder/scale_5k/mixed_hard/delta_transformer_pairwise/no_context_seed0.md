# Mixed hard Transformer pairwise audit: no_context

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard/mixed_hard_v0_10k_groups.npz`
- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard/delta_transformer/no_context_seed0/checkpoint.pt`
- mode: `no_context`
- test groups: `1000`
- top-1: `0.222000`
- mean rank: `2.918000`

| negative type | pairwise accuracy | mean margin | count |
| --- | ---: | ---: | ---: |
| same_state_diff_action | 0.530000 | 0.000952 | 2000 |
| same_action_diff_state | 0.511000 | 0.000179 | 2000 |

## Interpretation

`same_state_diff_action` tests whether the model uses action information.
`same_action_diff_state` tests whether the model uses state-dependent visual dynamics.

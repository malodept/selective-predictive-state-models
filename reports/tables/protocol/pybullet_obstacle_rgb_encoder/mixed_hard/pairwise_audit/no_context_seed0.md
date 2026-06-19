# Mixed hard pairwise audit: no_context

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz`
- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/delta_candidate_mlp/no_context_seed0/checkpoint.pt`
- mode: `no_context`
- test groups: `100`
- top-1: `0.270000`
- mean rank: `2.960000`

| negative type | pairwise accuracy | mean margin | count |
| --- | ---: | ---: | ---: |
| same_state_diff_action | 0.515000 | -0.001084 | 200 |
| same_action_diff_state | 0.505000 | -0.000678 | 200 |

## Interpretation

`same_state_diff_action` tests whether the model uses action information.
`same_action_diff_state` tests whether the model uses state-dependent visual dynamics.

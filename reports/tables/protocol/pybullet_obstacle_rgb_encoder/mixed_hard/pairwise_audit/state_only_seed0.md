# Mixed hard pairwise audit: state_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz`
- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/delta_candidate_mlp/state_only_seed0/checkpoint.pt`
- mode: `state_only`
- test groups: `100`
- top-1: `0.250000`
- mean rank: `2.830000`

| negative type | pairwise accuracy | mean margin | count |
| --- | ---: | ---: | ---: |
| same_state_diff_action | 0.560000 | 0.004776 | 200 |
| same_action_diff_state | 0.525000 | 0.001761 | 200 |

## Interpretation

`same_state_diff_action` tests whether the model uses action information.
`same_action_diff_state` tests whether the model uses state-dependent visual dynamics.

# Mixed hard pairwise audit: full

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz`
- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/delta_candidate_mlp/full_seed0/checkpoint.pt`
- mode: `full`
- test groups: `100`
- top-1: `0.390000`
- mean rank: `2.040000`

| negative type | pairwise accuracy | mean margin | count |
| --- | ---: | ---: | ---: |
| same_state_diff_action | 0.650000 | 0.030583 | 200 |
| same_action_diff_state | 0.830000 | 0.080335 | 200 |

## Interpretation

`same_state_diff_action` tests whether the model uses action information.
`same_action_diff_state` tests whether the model uses state-dependent visual dynamics.

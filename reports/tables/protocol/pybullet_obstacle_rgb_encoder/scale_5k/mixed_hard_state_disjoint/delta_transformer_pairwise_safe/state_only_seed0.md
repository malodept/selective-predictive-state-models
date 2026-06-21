# Safe mixed hard pairwise audit: state_only

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz`
- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/state_only_seed0/checkpoint.pt`
- mode: `state_only`
- split source: `explicit`
- test groups: `1000`
- top-1: `0.339000`
- mean rank: `2.139000`

| negative type | pairwise accuracy | mean margin | count |
| --- | ---: | ---: | ---: |
| same_action_diff_state | 0.746500 | 0.065267 | 2000 |
| same_state_diff_action | 0.503000 | 0.000669 | 2000 |

## Interpretation

This safe audit computes pairwise accuracy directly from the same distance matrix used for top-1 ranking.
Therefore, if top-1 is near-perfect, pairwise accuracy against each negative type should also be near-perfect.

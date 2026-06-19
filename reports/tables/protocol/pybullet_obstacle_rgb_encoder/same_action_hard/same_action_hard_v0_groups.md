# Same-action hard-negative group mining

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb/same_action_hard/same_action_hard_v0_groups.npz`
- groups requested: `1000`
- groups mined: `1000`
- candidates per group: `5`
- chance top-1: `0.200000`
- max position distance: `0.15`
- prefer opposite blocked: `True`
- anchor blocked fraction: `0.486000`
- offdiag blocked-mismatch fraction: `0.999750`

## Off-diagonal distributions

| quantity | mean | p05 | median | p95 |
| --- | ---: | ---: | ---: | ---: |
| start position distance | 0.046039 | 0.012849 | 0.045813 | 0.079512 |
| DINOv2 future distance | 38.675338 | 29.637166 | 38.233015 | 49.094644 |

## Interpretation

All candidates in a group share the same action. Therefore an action-only model should not be able to identify the correct future.
The groups are hard because off-diagonal candidates are selected from nearby starting positions and preferably opposite blocked/unblocked conditions.

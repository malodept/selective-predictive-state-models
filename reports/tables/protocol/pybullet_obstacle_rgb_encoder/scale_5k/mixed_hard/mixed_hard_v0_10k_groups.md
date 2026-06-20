# Mixed hard-negative group mining

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard/mixed_hard_v0_10k_groups.npz`
- groups requested: `10000`
- groups mined: `10000`
- candidates per group: `5`
- chance top-1: `0.200000`
- same-state different-action negatives: `2`
- same-action different-state negatives: `2`
- max position distance: `0.15`
- anchor blocked fraction: `0.500700`
- offdiag blocked-mismatch fraction: `0.812900`

## Off-diagonal distributions

| quantity | mean | p05 | median | p95 |
| --- | ---: | ---: | ---: | ---: |
| start position distance | 0.005271 | 0.000000 | 0.000062 | 0.017380 |
| DINOv2 delta distance | 25.572472 | 17.199694 | 25.651381 | 33.414197 |

## Interpretation

Each group mixes same-state/different-action negatives and same-action/different-state negatives.
This is designed so that action-only and state-only models should both fail, while a full state-action model should succeed.

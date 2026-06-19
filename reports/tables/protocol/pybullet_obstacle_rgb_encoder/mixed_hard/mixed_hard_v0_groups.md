# Mixed hard-negative group mining

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb/mixed_hard/mixed_hard_v0_groups.npz`
- groups requested: `1000`
- groups mined: `1000`
- candidates per group: `5`
- chance top-1: `0.200000`
- same-state different-action negatives: `2`
- same-action different-state negatives: `2`
- max position distance: `0.15`
- anchor blocked fraction: `0.486000`
- offdiag blocked-mismatch fraction: `0.802000`

## Off-diagonal distributions

| quantity | mean | p05 | median | p95 |
| --- | ---: | ---: | ---: | ---: |
| start position distance | 0.017410 | 0.000000 | 0.001043 | 0.058622 |
| DINOv2 delta distance | 25.973200 | 17.004080 | 26.177592 | 33.823589 |

## Interpretation

Each group mixes same-state/different-action negatives and same-action/different-state negatives.
This is designed so that action-only and state-only models should both fail, while a full state-action model should succeed.

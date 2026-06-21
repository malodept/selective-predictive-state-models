# State-disjoint mixed hard-negative group mining

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz`
- groups total: `10000`
- groups by split: `{'train': 8000, 'val': 1000, 'test': 1000}`
- states by split: `{'train': 4000, 'val': 500, 'test': 500}`
- candidates per group: `5`
- chance top-1: `0.200000`
- same-state different-action negatives: `2`
- same-action different-state negatives: `2`
- max position distance: `0.15`
- prefer blocked mismatch: `True`

## State-overlap verification

| split pair | state overlap |
| --- | ---: |
| train/val | 0 |
| train/test | 0 |
| val/test | 0 |

## Off-diagonal diagnostics

| quantity | value |
| --- | ---: |
| mean start-position distance | 0.037911 |
| median start-position distance | 0.000308 |
| p95 start-position distance | 0.119606 |
| blocked-mismatch fraction | 0.703575 |

## Interpretation

This group file enforces state-disjoint train/validation/test splits before hard-negative mining.
It is the stronger generalization protocol: no simulator state used by candidate futures in train appears in validation or test.

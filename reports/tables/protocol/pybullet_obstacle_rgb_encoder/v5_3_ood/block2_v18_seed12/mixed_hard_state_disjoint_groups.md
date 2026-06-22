# State-disjoint mixed hard-negative group mining

- data: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_v18_seed12/pybullet_obstacle_rgb_v1_ood_block2_v18_seed12_dinov2_vits14_pool4.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_v18_seed12/mixed_hard_state_disjoint_v0_groups.npz`
- groups total: `4000`
- groups by split: `{'train': 3000, 'val': 500, 'test': 500}`
- states by split: `{'train': 1600, 'val': 200, 'test': 200}`
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
| mean start-position distance | 0.047718 |
| median start-position distance | 0.000748 |
| p95 start-position distance | 0.141140 |
| blocked-mismatch fraction | 0.704063 |

## Interpretation

This group file enforces state-disjoint train/validation/test splits before hard-negative mining.
It is the stronger generalization protocol: no simulator state used by candidate futures in train appears in validation or test.

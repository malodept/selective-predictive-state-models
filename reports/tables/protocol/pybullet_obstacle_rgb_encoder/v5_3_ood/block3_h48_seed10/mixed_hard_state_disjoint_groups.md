# State-disjoint mixed hard-negative group mining

- data: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h48_seed10/pybullet_obstacle_rgb_v1_ood_block3_h48_seed10_dinov2_vits14_pool4.npz`
- output: `10000`
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
| mean start-position distance | 0.047323 |
| median start-position distance | 0.000934 |
| p95 start-position distance | 0.139283 |
| blocked-mismatch fraction | 0.690750 |

## Interpretation

This group file enforces state-disjoint train/validation/test splits before hard-negative mining.
It is the stronger generalization protocol: no simulator state used by candidate futures in train appears in validation or test.

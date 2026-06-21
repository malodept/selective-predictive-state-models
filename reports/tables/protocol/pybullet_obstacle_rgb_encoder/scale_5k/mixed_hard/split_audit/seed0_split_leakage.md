# Mixed hard split leakage audit

- data: `outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz`
- groups: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard/mixed_hard_v0_10k_groups.npz`
- split seed: `0`
- mined groups: `10000`

## Split content

| split | groups | samples used | states used | anchor states | candidate states |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 8000 | 20142 | 4963 | 4364 | 4963 |
| val | 1000 | 4547 | 2274 | 919 | 2274 |
| test | 1000 | 4544 | 2239 | 930 | 2239 |

## Overlap between splits

| split pair | sample overlap | anchor sample overlap | candidate sample overlap | state overlap | anchor state overlap | candidate state overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train/val | 3661 | 0 | 3661 | 2261 | 722 | 2261 |
| train/test | 3627 | 0 | 3627 | 2223 | 737 | 2223 |
| val/test | 855 | 0 | 855 | 1036 | 145 | 1036 |

## Interpretation

This audit checks whether the random mined-group split reuses exact-intervention samples or simulator states across train/validation/test.
For the strongest generalization claim, the ideal protocol is state-disjoint: no simulator state should appear in more than one split.
If state overlap is high, the current result remains a strong protocol/architecture result, but the next benchmark should enforce state-disjoint mining.

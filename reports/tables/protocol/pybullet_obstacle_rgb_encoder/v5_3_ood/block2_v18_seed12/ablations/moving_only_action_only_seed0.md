# Moving-only tie-aware mixed hard evaluation: action_only

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/action_only_seed0/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `394`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.324873 |
| strict top-1 | 0.324873 |
| tie-aware top-1 | 0.324873 |
| mean tie count at minimum | 1.000000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.486041 | 0.000000 | 0.513959 | 0.486041 | -0.002533 | 788 |
| same_state_diff_action | 0.881980 | 0.000000 | 0.118020 | 0.881980 | 0.056950 | 788 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

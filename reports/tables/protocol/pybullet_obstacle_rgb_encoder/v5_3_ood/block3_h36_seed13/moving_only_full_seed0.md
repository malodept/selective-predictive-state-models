# Moving-only tie-aware mixed hard evaluation: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed0/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `409`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.828851 |
| strict top-1 | 0.828851 |
| tie-aware top-1 | 0.828851 |
| mean tie count at minimum | 1.000000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.863081 | 0.000000 | 0.136919 | 0.863081 | 0.145654 | 818 |
| same_state_diff_action | 0.979218 | 0.000000 | 0.020782 | 0.979218 | 0.142981 | 818 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

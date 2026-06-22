# Moving-only tie-aware mixed hard evaluation: state_only

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/state_only_seed0/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `394`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.332487 |
| strict top-1 | 0.332487 |
| tie-aware top-1 | 0.332487 |
| mean tie count at minimum | 1.000000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.898477 | 0.000000 | 0.101523 | 0.898477 | 0.071141 | 788 |
| same_state_diff_action | 0.486041 | 0.000000 | 0.513959 | 0.486041 | -0.001080 | 788 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

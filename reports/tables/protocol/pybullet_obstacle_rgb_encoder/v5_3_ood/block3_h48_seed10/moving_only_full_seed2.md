# Moving-only tie-aware mixed hard evaluation: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed2/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `402`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.875622 |
| strict top-1 | 0.875622 |
| tie-aware top-1 | 0.875622 |
| mean tie count at minimum | 1.000000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.901741 | 0.000000 | 0.098259 | 0.901741 | 0.166312 | 804 |
| same_state_diff_action | 0.993781 | 0.000000 | 0.006219 | 0.993781 | 0.150226 | 804 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

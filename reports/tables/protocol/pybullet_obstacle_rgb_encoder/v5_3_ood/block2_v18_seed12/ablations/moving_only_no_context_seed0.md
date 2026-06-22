# Moving-only tie-aware mixed hard evaluation: no_context

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/no_context_seed0/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `394`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.142132 |
| strict top-1 | 0.142132 |
| tie-aware top-1 | 0.142132 |
| mean tie count at minimum | 1.000000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.484772 | 0.000000 | 0.515228 | 0.484772 | -0.001351 | 788 |
| same_state_diff_action | 0.392132 | 0.000000 | 0.607868 | 0.392132 | -0.007641 | 788 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

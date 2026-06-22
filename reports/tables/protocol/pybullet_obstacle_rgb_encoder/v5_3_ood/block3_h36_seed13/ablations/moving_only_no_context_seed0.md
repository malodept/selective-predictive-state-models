# Moving-only tie-aware mixed hard evaluation: no_context

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/no_context_seed0/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `409`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.185819 |
| strict top-1 | 0.185819 |
| tie-aware top-1 | 0.185819 |
| mean tie count at minimum | 1.000000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.513447 | 0.000000 | 0.486553 | 0.513447 | 0.000809 | 818 |
| same_state_diff_action | 0.492665 | 0.000000 | 0.507335 | 0.492665 | -0.000577 | 818 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

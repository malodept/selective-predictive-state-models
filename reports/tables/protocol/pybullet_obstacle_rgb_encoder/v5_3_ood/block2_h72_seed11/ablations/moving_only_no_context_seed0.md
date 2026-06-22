# Moving-only tie-aware mixed hard evaluation: no_context

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/no_context_seed0/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `416`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.201923 |
| strict top-1 | 0.199519 |
| tie-aware top-1 | 0.200721 |
| mean tie count at minimum | 1.004808 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.503606 | 0.000000 | 0.496394 | 0.503606 | 0.001203 | 832 |
| same_state_diff_action | 0.450721 | 0.003606 | 0.545673 | 0.454327 | -0.004008 | 832 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

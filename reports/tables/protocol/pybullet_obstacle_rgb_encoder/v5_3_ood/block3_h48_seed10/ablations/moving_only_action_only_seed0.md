# Moving-only tie-aware mixed hard evaluation: action_only

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/action_only_seed0/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `402`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.261194 |
| strict top-1 | 0.261194 |
| tie-aware top-1 | 0.261194 |
| mean tie count at minimum | 1.000000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.391791 | 0.000000 | 0.608209 | 0.391791 | -0.009375 | 804 |
| same_state_diff_action | 0.936567 | 0.000000 | 0.063433 | 0.936567 | 0.049707 | 804 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

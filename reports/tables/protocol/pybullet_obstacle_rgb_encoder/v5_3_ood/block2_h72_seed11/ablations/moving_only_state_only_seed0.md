# Moving-only tie-aware mixed hard evaluation: state_only

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/state_only_seed0/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `416`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.348558 |
| strict top-1 | 0.348558 |
| tie-aware top-1 | 0.348558 |
| mean tie count at minimum | 1.000000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.865385 | 0.000000 | 0.134615 | 0.865385 | 0.066344 | 832 |
| same_state_diff_action | 0.514423 | 0.003606 | 0.481971 | 0.518029 | 0.001964 | 832 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

# Moving-only tie-aware mixed hard evaluation: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed1/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `413`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.832930 |
| strict top-1 | 0.832930 |
| tie-aware top-1 | 0.832930 |
| mean tie count at minimum | 1.000000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.869249 | 0.000000 | 0.130751 | 0.869249 | 0.125578 | 826 |
| same_state_diff_action | 0.985472 | 0.000000 | 0.014528 | 0.985472 | 0.140148 | 826 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

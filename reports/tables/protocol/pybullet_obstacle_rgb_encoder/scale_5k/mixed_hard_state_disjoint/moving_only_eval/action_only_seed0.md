# Moving-only tie-aware mixed hard evaluation: action_only

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/action_only_seed0/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `819`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.327228 |
| strict top-1 | 0.327228 |
| tie-aware top-1 | 0.327228 |
| mean tie count at minimum | 1.000000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.495726 | 0.000000 | 0.504274 | 0.495726 | -0.000046 | 1638 |
| same_state_diff_action | 0.902930 | 0.000000 | 0.097070 | 0.902930 | 0.057131 | 1638 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

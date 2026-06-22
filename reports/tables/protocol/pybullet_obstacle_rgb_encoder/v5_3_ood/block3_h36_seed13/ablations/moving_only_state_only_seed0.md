# Moving-only tie-aware mixed hard evaluation: state_only

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/state_only_seed0/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `409`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.337408 |
| strict top-1 | 0.337408 |
| tie-aware top-1 | 0.337408 |
| mean tie count at minimum | 1.000000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.898533 | 0.000000 | 0.101467 | 0.898533 | 0.063702 | 818 |
| same_state_diff_action | 0.506112 | 0.000000 | 0.493888 | 0.506112 | -0.001129 | 818 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

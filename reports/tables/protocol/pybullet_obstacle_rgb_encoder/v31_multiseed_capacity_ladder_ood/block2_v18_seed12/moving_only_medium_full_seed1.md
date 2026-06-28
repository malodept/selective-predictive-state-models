# Moving-only tie-aware mixed hard evaluation: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/medium_full_seed1/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `394`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.989848 |
| strict top-1 | 0.989848 |
| tie-aware top-1 | 0.989848 |
| mean tie count at minimum | 1.000000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.993655 | 0.000000 | 0.006345 | 0.993655 | 0.218740 | 788 |
| same_state_diff_action | 0.998731 | 0.000000 | 0.001269 | 0.998731 | 0.198138 | 788 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

# Moving-only tie-aware mixed hard evaluation: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed2/checkpoint.pt`
- split source: `explicit`
- excluded action: `stay`
- moving-only test groups: `416`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.975962 |
| strict top-1 | 0.973558 |
| tie-aware top-1 | 0.974760 |
| mean tie count at minimum | 1.002404 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.983173 | 0.000000 | 0.016827 | 0.983173 | 0.189935 | 832 |
| same_state_diff_action | 0.985577 | 0.003606 | 0.010817 | 0.989183 | 0.195729 | 832 |

## Interpretation

This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.
It reports performance on the identifiable moving-action subset.

# Tie-aware mixed hard evaluation: no_context

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/no_context_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `1000`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.263000 |
| strict top-1 | 0.157000 |
| tie-aware top-1 | 0.192333 |
| correct tied with another candidate | 0.106000 |
| mean tie count at minimum | 1.212000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.434500 | 0.181000 | 0.384500 | 0.615500 | 0.000884 | 2000 |
| same_state_diff_action | 0.499500 | 0.000000 | 0.500500 | 0.499500 | -0.000126 | 2000 |

## Interpretation

The biased argmin top-1 can be inflated when the correct candidate is placed in column 0 and another candidate has exactly the same distance.
The strict top-1 and tie-aware top-1 are therefore more reliable for reporting generalization.

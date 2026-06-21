# Tie-aware mixed hard evaluation: full

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `1000`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.999000 |
| strict top-1 | 0.818000 |
| tie-aware top-1 | 0.878333 |
| correct tied with another candidate | 0.181000 |
| mean tie count at minimum | 1.362000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.819000 | 0.181000 | 0.000000 | 1.000000 | 0.189139 | 2000 |
| same_state_diff_action | 0.999500 | 0.000000 | 0.000500 | 0.999500 | 0.191758 | 2000 |

## Interpretation

The biased argmin top-1 can be inflated when the correct candidate is placed in column 0 and another candidate has exactly the same distance.
The strict top-1 and tie-aware top-1 are therefore more reliable for reporting generalization.

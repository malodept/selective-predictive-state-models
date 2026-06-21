# Tie-aware mixed hard evaluation: state_only

- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/state_only_seed0/checkpoint.pt`
- split source: `explicit`
- test groups: `1000`
- epsilon: `1e-08`

## Global metrics

| metric | value |
| --- | ---: |
| biased argmin top-1 | 0.339000 |
| strict top-1 | 0.281000 |
| tie-aware top-1 | 0.300333 |
| correct tied with another candidate | 0.058000 |
| mean tie count at minimum | 1.116000 |

## Pairwise margin decomposition

| negative type | strict win | tie | loss | non-loss | mean margin | count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| same_action_diff_state | 0.746500 | 0.181000 | 0.072500 | 0.927500 | 0.065267 | 2000 |
| same_state_diff_action | 0.503000 | 0.000000 | 0.497000 | 0.503000 | 0.000669 | 2000 |

## Interpretation

The biased argmin top-1 can be inflated when the correct candidate is placed in column 0 and another candidate has exactly the same distance.
The strict top-1 and tie-aware top-1 are therefore more reliable for reporting generalization.

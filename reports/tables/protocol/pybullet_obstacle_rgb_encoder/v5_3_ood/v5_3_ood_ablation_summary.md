# SPSM v5.3 OOD ablation summary

| variant | full top-1 | action-only top-1 | state-only top-1 | no-context top-1 | action-only same-state/diff-action | state-only same-action/diff-state |
|---|---:|---:|---:|---:|---:|---:|
| block2_h72_seed11 | 0.972756 ± 0.006049 | 0.350962 | 0.348558 | 0.199519 | 0.925481 | 0.865385 |
| block2_v18_seed12 | 0.992386 ± 0.002538 | 0.324873 | 0.332487 | 0.142132 | 0.881980 | 0.898477 |
| block3_h36_seed13 | 0.854931 ± 0.023747 | 0.322738 | 0.337408 | 0.185819 | 0.952323 | 0.898533 |
| block3_h48_seed10 | 0.872305 ± 0.029989 | 0.261194 | 0.305970 | 0.181592 | 0.936567 | 0.861940 |
| block3_h72_seed14 | 0.807910 ± 0.010918 | 0.263923 | 0.326877 | 0.196126 | 0.945521 | 0.828087 |

## Interpretation

The ablations reveal a factorized structure. The action-only model is strong on same-state/different-action negatives but weak on same-action/different-state negatives. The state-only model shows the opposite pattern. The no-context model stays close to chance. The full model combines both axes, confirming that the state-disjoint SPSM result is not explained by a single shortcut. Under block3 OOD shifts, the main degradation is therefore best interpreted as a state-geometry or state-action-interaction difficulty rather than a failure of action grounding alone.

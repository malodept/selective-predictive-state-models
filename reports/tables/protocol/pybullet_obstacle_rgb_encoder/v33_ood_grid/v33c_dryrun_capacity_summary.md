# SPSM v33C dry-run OOD grid summary

This summarizes the two dry-run OOD-grid variants before launching the full mini-grid.

## Capacity results

| variant | Tiny | Small | Medium | Full | best | range |
|---|---:|---:|---:|---:|---|---:|
| block1_h36_seed20 | 0.935129 | 0.926561 | 0.970624 | 0.968176 | `Medium` | 0.044063 |
| block4_h72_seed20 | 0.581907 | 0.574572 | 0.570905 | 0.534230 | `Tiny` | 0.047677 |

## Decomposition

| variant | model | same-action/diff-state | same-state/diff-action |
|---|---|---:|---:|
| block1_h36_seed20 | Tiny | 0.954712 | 0.995104 |
| block1_h36_seed20 | Small | 0.947980 | 0.993268 |
| block1_h36_seed20 | Medium | 0.981640 | 0.998776 |
| block1_h36_seed20 | Full | 0.979192 | 0.998164 |
| block4_h72_seed20 | Tiny | 0.745721 | 0.948044 |
| block4_h72_seed20 | Small | 0.743888 | 0.943154 |
| block4_h72_seed20 | Medium | 0.759169 | 0.926039 |
| block4_h72_seed20 | Full | 0.742054 | 0.882641 |

## Interpretation

- `block1_h36_seed20` is easy and near-saturated.
- `block4_h72_seed20` is valid but hard, with no evidence of massive tie degeneracy.
- This supports launching a systematic blocked/horizon/layout grid.

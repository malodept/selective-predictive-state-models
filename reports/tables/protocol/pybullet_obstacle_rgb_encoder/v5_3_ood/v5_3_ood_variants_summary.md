# SPSM v5.3 OOD variants summary

| variant | moving-only strict top-1 | same-action/diff-state | same-state/diff-action | full strict top-1 | full tie-aware top-1 | ECE tie-aware | strict @80% cov | strict @60% cov | strict @50% cov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| block2_h72_seed11 | 0.972756 ± 0.006049 | 0.987580 ± 0.002502 | 0.985978 ± 0.002502 | 0.809333 ± 0.005033 | 0.866333 ± 0.005033 | 0.014928 ± 0.002837 | 0.995000 ± 0.002500 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| block2_v18_seed12 | 0.992386 ± 0.002538 | 0.996616 ± 0.002642 | 0.998731 ± 0.000000 | 0.782000 ± 0.002000 | 0.852667 ± 0.002000 | 0.009360 ± 0.001392 | 0.977500 ± 0.002500 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 |
| block3_h36_seed13 | 0.854931 ± 0.023747 | 0.884271 ± 0.019342 | 0.982885 ± 0.003668 | 0.699333 ± 0.019425 | 0.760000 ± 0.019425 | 0.042221 ± 0.017215 | 0.865000 ± 0.022220 | 0.966667 ± 0.016667 | 0.988000 ± 0.008000 |
| block3_h48_seed10 | 0.872305 ± 0.029989 | 0.903399 ± 0.024917 | 0.992952 ± 0.000718 | 0.701333 ± 0.024111 | 0.766222 ± 0.024151 | 0.042017 ± 0.024000 | 0.874167 ± 0.030139 | 0.975555 ± 0.003849 | 0.992000 ± 0.006928 |
| block3_h72_seed14 | 0.807910 ± 0.010918 | 0.847054 ± 0.014444 | 0.983051 ± 0.005548 | 0.667333 ± 0.009018 | 0.725333 ± 0.009018 | 0.063863 ± 0.000872 | 0.821667 ± 0.011547 | 0.937778 ± 0.018954 | 0.974667 ± 0.019732 |

## Main interpretation

The frozen v5.2 state-action Transformer remains highly robust to horizon-only and velocity-only shifts, but degrades substantially when the number of blocked directions increases from 2 to 3. The dominant failure mode is same-action/different-state discrimination, while same-state/different-action discrimination remains much stronger. This suggests that the model keeps action grounding but struggles with more constrained environment geometry. Confidence remains useful for selective prediction: high-confidence subsets recover much stronger strict top-1 under OOD shift.

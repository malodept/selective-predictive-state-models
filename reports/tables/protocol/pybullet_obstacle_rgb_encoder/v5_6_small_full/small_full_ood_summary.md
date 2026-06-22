# SPSM v5.6 small-full OOD evaluation

The small-full model is a reduced full state-action Transformer: model_dim=128, layers=1, heads=4.
It is compared against the v5.2 full Transformer across controlled OOD variants.

| variant | full top-1 mean seeds 0-2 | full seed0 top-1 | small-full seed0 top-1 | gap full seed0 - small | small same-action/diff-state | small same-state/diff-action |
|---|---:|---:|---:|---:|---:|---:|
| block2_h72_seed11 | 0.972756 ± 0.006049 | 0.973558 | 0.927885 | 0.045673 | 0.953125 | 0.985577 |
| block2_v18_seed12 | 0.992386 ± 0.002538 | 0.994924 | 0.987310 | 0.007614 | 0.992386 | 0.998731 |
| block3_h36_seed13 | 0.854931 ± 0.023747 | 0.828851 | 0.897311 | -0.068460 | 0.936430 | 0.995110 |
| block3_h48_seed10 | 0.872305 ± 0.029989 | 0.840796 | 0.853234 | -0.012438 | 0.896766 | 0.997512 |
| block3_h72_seed14 | 0.807910 ± 0.010918 | 0.818402 | 0.731235 | 0.087167 | 0.786925 | 0.991525 |

## Interpretation

If small-full remains close to full on easy OOD but drops more on block3 variants, it is a strong cheap/expensive setup for value-of-computation. If small-full is indistinguishable from full everywhere, then it is already sufficient and the expensive model is not justified in this benchmark.

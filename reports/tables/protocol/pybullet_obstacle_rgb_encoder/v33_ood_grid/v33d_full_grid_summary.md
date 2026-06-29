# SPSM v33D full OOD mini-grid summary

This summarizes the systematic OOD grid: blocked directions `{1,2,3,4}`, horizons `{36,72}`, layout seeds `{20,21,22}`, and four seed0 capacities.

## Mean tie-aware top-1 over layout seeds

| horizon | blocked | Tiny | Small | Medium | Full | best mean | range |
|---:|---:|---:|---:|---:|---:|---|---:|
| 36 | 1 | 0.941±0.006 | 0.940±0.013 | 0.971±0.005 | 0.966±0.003 | `Medium` | 0.032 |
| 36 | 2 | 0.996±0.006 | 0.996±0.003 | 0.995±0.004 | 0.998±0.001 | `Full` | 0.004 |
| 36 | 3 | 0.915±0.010 | 0.911±0.013 | 0.927±0.003 | 0.869±0.008 | `Medium` | 0.058 |
| 36 | 4 | 0.589±0.009 | 0.602±0.025 | 0.579±0.012 | 0.543±0.012 | `Small` | 0.059 |
| 72 | 1 | 0.881±0.002 | 0.906±0.015 | 0.927±0.004 | 0.879±0.008 | `Medium` | 0.048 |
| 72 | 2 | 0.974±0.005 | 0.934±0.010 | 0.970±0.004 | 0.977±0.004 | `Full` | 0.043 |
| 72 | 3 | 0.821±0.014 | 0.737±0.009 | 0.839±0.014 | 0.813±0.010 | `Medium` | 0.102 |
| 72 | 4 | 0.589±0.009 | 0.602±0.025 | 0.579±0.012 | 0.543±0.012 | `Small` | 0.059 |

## State/action decomposition

| horizon | blocked | model | same-action/diff-state | same-state/diff-action |
|---:|---:|---|---:|---:|
| 36 | 1 | Tiny | 0.959±0.005 | 0.996±0.001 |
| 36 | 1 | Small | 0.961±0.011 | 0.996±0.003 |
| 36 | 1 | Medium | 0.984±0.004 | 0.997±0.002 |
| 36 | 1 | Full | 0.977±0.003 | 0.998±0.003 |
| 36 | 2 | Tiny | 0.998±0.004 | 1.000±0.000 |
| 36 | 2 | Small | 0.998±0.002 | 0.999±0.000 |
| 36 | 2 | Medium | 0.997±0.002 | 0.999±0.000 |
| 36 | 2 | Full | 0.999±0.001 | 1.000±0.000 |
| 36 | 3 | Tiny | 0.944±0.002 | 0.995±0.001 |
| 36 | 3 | Small | 0.943±0.008 | 0.992±0.003 |
| 36 | 3 | Medium | 0.957±0.003 | 0.989±0.003 |
| 36 | 3 | Full | 0.904±0.007 | 0.984±0.003 |
| 36 | 4 | Tiny | 0.746±0.010 | 0.949±0.004 |
| 36 | 4 | Small | 0.762±0.016 | 0.950±0.006 |
| 36 | 4 | Medium | 0.761±0.004 | 0.912±0.013 |
| 36 | 4 | Full | 0.746±0.004 | 0.891±0.009 |
| 72 | 1 | Tiny | 0.923±0.005 | 0.977±0.002 |
| 72 | 1 | Small | 0.938±0.012 | 0.985±0.007 |
| 72 | 1 | Medium | 0.960±0.006 | 0.981±0.004 |
| 72 | 1 | Full | 0.935±0.009 | 0.964±0.004 |
| 72 | 2 | Tiny | 0.984±0.005 | 0.993±0.001 |
| 72 | 2 | Small | 0.956±0.004 | 0.992±0.003 |
| 72 | 2 | Medium | 0.985±0.001 | 0.994±0.003 |
| 72 | 2 | Full | 0.989±0.004 | 0.990±0.001 |
| 72 | 3 | Tiny | 0.866±0.008 | 0.990±0.003 |
| 72 | 3 | Small | 0.793±0.014 | 0.988±0.002 |
| 72 | 3 | Medium | 0.887±0.008 | 0.984±0.006 |
| 72 | 3 | Full | 0.846±0.006 | 0.979±0.002 |
| 72 | 4 | Tiny | 0.746±0.010 | 0.949±0.004 |
| 72 | 4 | Small | 0.762±0.016 | 0.950±0.006 |
| 72 | 4 | Medium | 0.761±0.004 | 0.912±0.013 |
| 72 | 4 | Full | 0.746±0.004 | 0.891±0.009 |

## Best capacity per individual variant

| variant | Tiny | Small | Medium | Full | best |
|---|---:|---:|---:|---:|---|
| block1_h36_seed20 | 0.935 | 0.927 | 0.971 | 0.968 | `Medium` |
| block1_h36_seed21 | 0.948 | 0.952 | 0.976 | 0.963 | `Medium` |
| block1_h36_seed22 | 0.940 | 0.941 | 0.967 | 0.966 | `Medium` |
| block1_h72_seed20 | 0.879 | 0.892 | 0.924 | 0.879 | `Medium` |
| block1_h72_seed21 | 0.883 | 0.922 | 0.926 | 0.887 | `Medium` |
| block1_h72_seed22 | 0.880 | 0.903 | 0.932 | 0.872 | `Medium` |
| block2_h36_seed20 | 0.999 | 0.999 | 0.997 | 1.000 | `Full` |
| block2_h36_seed21 | 1.000 | 0.996 | 0.996 | 0.997 | `Tiny` |
| block2_h36_seed22 | 0.989 | 0.993 | 0.990 | 0.998 | `Full` |
| block2_h72_seed20 | 0.973 | 0.924 | 0.967 | 0.975 | `Full` |
| block2_h72_seed21 | 0.979 | 0.936 | 0.975 | 0.982 | `Full` |
| block2_h72_seed22 | 0.969 | 0.943 | 0.969 | 0.975 | `Full` |
| block3_h36_seed20 | 0.922 | 0.911 | 0.930 | 0.867 | `Medium` |
| block3_h36_seed21 | 0.903 | 0.898 | 0.923 | 0.862 | `Medium` |
| block3_h36_seed22 | 0.920 | 0.923 | 0.928 | 0.877 | `Medium` |
| block3_h72_seed20 | 0.831 | 0.746 | 0.839 | 0.815 | `Medium` |
| block3_h72_seed21 | 0.805 | 0.735 | 0.825 | 0.803 | `Medium` |
| block3_h72_seed22 | 0.825 | 0.729 | 0.853 | 0.822 | `Medium` |
| block4_h36_seed20 | 0.582 | 0.575 | 0.571 | 0.534 | `Tiny` |
| block4_h36_seed21 | 0.586 | 0.610 | 0.574 | 0.538 | `Small` |
| block4_h36_seed22 | 0.599 | 0.622 | 0.593 | 0.557 | `Small` |
| block4_h72_seed20 | 0.582 | 0.575 | 0.571 | 0.534 | `Tiny` |
| block4_h72_seed21 | 0.586 | 0.610 | 0.574 | 0.538 | `Small` |
| block4_h72_seed22 | 0.599 | 0.622 | 0.593 | 0.557 | `Small` |

## Key diagnostics


### Horizon 36

- blocked=1: best=Medium, Medium-Full gap=0.0058.
- blocked=2: best=Full, Medium-Full gap=-0.0037.
- blocked=3: best=Medium, Medium-Full gap=0.0583.
- blocked=4: best=Small, Medium-Full gap=0.0361.

### Horizon 72

- blocked=1: best=Medium, Medium-Full gap=0.0480.
- blocked=2: best=Full, Medium-Full gap=-0.0071.
- blocked=3: best=Medium, Medium-Full gap=0.0259.
- blocked=4: best=Small, Medium-Full gap=0.0361.

## Interpretation guide

- If performance decreases monotonically with blocked directions, the grid validates constrained geometry as a systematic difficulty axis.
- If Full is not consistently best across blocked/horizon cells, the non-monotonic capacity result generalizes beyond hand-picked OOD shifts.
- If the same-action/diff-state term degrades faster than same-state/diff-action, the main failure remains state discrimination under fixed action.

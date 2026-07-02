# SPSM v47 Compute Regime Map

This analysis turns the multi-capacity predictions into query-level compute regimes. It asks not only which capacity is best on average, but whether additional predictive compute is useful, harmful, unnecessary, or insufficient for each OOD query.

## Inputs

- predictions: `reports/tables/protocol/pybullet_obstacle_rgb_encoder/v32_multiseed_multicapacity_routing/multiseed_multicapacity_predictions.csv`
- router results: `reports/tables/protocol/pybullet_obstacle_rgb_encoder/v39_margin_gated_router/v39_margin_gated_router_loso_results.csv`

## Regime definitions

- `all_correct`: Tiny, Small, Medium, and Full are all correct; extra compute is unnecessary.
- `none_correct`: no capacity is correct; current compute ladder cannot solve the query.
- `useful_compute`: Tiny is wrong but at least one larger capacity is correct.
- `anti_rescue`: Tiny is correct but Full is wrong; a monotone fallback to Full would hurt.
- `tiny_only`, `medium_only`, `full_only`: only that capacity is correct.
- pattern order is `Tiny Small Medium Full`.

## Hard-OOD compute-regime rates

| shift | n | all correct | none correct | useful compute | anti-rescue | Tiny wrong / other correct | Tiny correct / Full wrong | Full correct / Tiny wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3-block, H=36 | 1227 | 0.768 | 0.053 | 0.049 | 0.052 | 0.071 | 0.067 | 0.046 |
| 3-block, H=48 | 1206 | 0.760 | 0.051 | 0.061 | 0.044 | 0.096 | 0.051 | 0.070 |
| 3-block, H=72 | 1239 | 0.655 | 0.065 | 0.084 | 0.065 | 0.126 | 0.089 | 0.087 |

## Dominant correctness patterns

| shift | pattern | count | frac |
|---|---|---:|---:|
| 3-block, H=36 | `1111` | 942 | 0.768 |
| 3-block, H=36 | `0000` | 65 | 0.053 |
| 3-block, H=36 | `1110` | 33 | 0.027 |
| 3-block, H=36 | `1011` | 30 | 0.024 |
| 3-block, H=36 | `0111` | 25 | 0.020 |
| 3-block, H=36 | `1100` | 23 | 0.019 |
| 3-block, H=48 | `1111` | 916 | 0.760 |
| 3-block, H=48 | `0000` | 61 | 0.051 |
| 3-block, H=48 | `0111` | 44 | 0.036 |
| 3-block, H=48 | `1110` | 29 | 0.024 |
| 3-block, H=48 | `1011` | 28 | 0.023 |
| 3-block, H=48 | `1101` | 21 | 0.017 |
| 3-block, H=72 | `1111` | 812 | 0.655 |
| 3-block, H=72 | `0000` | 80 | 0.065 |
| 3-block, H=72 | `1011` | 48 | 0.039 |
| 3-block, H=72 | `0011` | 47 | 0.038 |
| 3-block, H=72 | `1110` | 39 | 0.031 |
| 3-block, H=72 | `0111` | 39 | 0.031 |

## Oracle headroom by λ

| shift | λ | best fixed | fixed util | oracle util | oracle headroom |
|---|---:|---|---:|---:|---:|
| 3-block, H=36 | 0.00 | `Medium` | 0.879 | 0.947 | 0.068 |
| 3-block, H=36 | 0.02 | `Tiny` | 0.872 | 0.943 | 0.071 |
| 3-block, H=36 | 0.05 | `Tiny` | 0.867 | 0.937 | 0.070 |
| 3-block, H=36 | 0.10 | `Tiny` | 0.858 | 0.928 | 0.070 |
| 3-block, H=36 | 0.20 | `Tiny` | 0.839 | 0.908 | 0.069 |
| 3-block, H=36 | 0.30 | `Tiny` | 0.820 | 0.889 | 0.068 |
| 3-block, H=48 | 0.00 | `Medium` | 0.884 | 0.949 | 0.066 |
| 3-block, H=48 | 0.02 | `Medium` | 0.876 | 0.945 | 0.069 |
| 3-block, H=48 | 0.05 | `Medium` | 0.865 | 0.939 | 0.074 |
| 3-block, H=48 | 0.10 | `Small` | 0.847 | 0.929 | 0.082 |
| 3-block, H=48 | 0.20 | `Small` | 0.827 | 0.909 | 0.081 |
| 3-block, H=48 | 0.30 | `Small` | 0.807 | 0.888 | 0.081 |
| 3-block, H=72 | 0.00 | `Medium` | 0.829 | 0.935 | 0.107 |
| 3-block, H=72 | 0.02 | `Medium` | 0.821 | 0.931 | 0.110 |
| 3-block, H=72 | 0.05 | `Medium` | 0.810 | 0.925 | 0.115 |
| 3-block, H=72 | 0.10 | `Tiny` | 0.791 | 0.915 | 0.124 |
| 3-block, H=72 | 0.20 | `Tiny` | 0.772 | 0.894 | 0.122 |
| 3-block, H=72 | 0.30 | `Tiny` | 0.754 | 0.873 | 0.120 |

## Compute recommendations

| shift | useful | anti-rescue | mean oracle gap | stable router λ count | recommendation |
|---|---:|---:|---:|---:|---|
| 3-block, H=36 | 0.049 | 0.052 | 0.070 | 0 | use fixed best capacity; routing optional |
| 3-block, H=48 | 0.061 | 0.044 | 0.080 | 0 | use fixed best capacity; routing optional |
| 3-block, H=72 | 0.084 | 0.065 | 0.120 | 0 | use fixed best capacity; routing optional |

## Interpretation

- `useful_compute` is the deployable reason to leave Tiny.
- `anti_rescue` shows why larger models are not a monotone fallback.
- Oracle headroom measures available capacity-selection value; router gain measures how much of it is recoverable from observable signals.
- The map converts value-of-computation from a scalar average into a regime-level diagnostic.

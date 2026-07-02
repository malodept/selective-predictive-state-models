# SPSM v47B Clean Compute Regime Map

This corrected version uses exhaustive query-level regimes. Useful compute is defined as `Tiny wrong and at least one other capacity correct`; anti-rescue is defined as `Tiny correct and Full wrong`.

## Hard-OOD compute-regime rates

| shift | n | all | none | useful | anti-rescue | other | sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3-block, H=36 | 1227 | 0.768 | 0.053 | 0.071 | 0.067 | 0.042 | 1.000 |
| 3-block, H=48 | 1206 | 0.760 | 0.051 | 0.096 | 0.051 | 0.042 | 1.000 |
| 3-block, H=72 | 1239 | 0.655 | 0.065 | 0.126 | 0.089 | 0.065 | 1.000 |

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

| shift | λ | best fixed | fixed util | oracle util | oracle gap |
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

## Compute-regime recommendations

| shift | useful | anti-rescue | oracle gap | stable router λ | recommendation |
|---|---:|---:|---:|---:|---|
| 3-block, H=36 | 0.071 | 0.067 | 0.070 | 1 | mostly fixed-capacity regime; limited routing value |
| 3-block, H=48 | 0.096 | 0.051 | 0.080 | 4 | deploy margin-gated routing; useful signal is routable |
| 3-block, H=72 | 0.126 | 0.089 | 0.120 | 2 | high oracle value; current routing only partially recovers it |

## Interpretation

- H=48 is the cleanest routable regime: useful compute is higher than H=36, anti-rescue is lower than H=72, and v39 gives the most stable router gains.
- H=72 has the largest oracle headroom and useful-compute rate, but also the largest anti-rescue rate, explaining why monotone deferral is unsafe and routing remains partial.
- H=36 has limited routing value because most queries are already all-correct and useful-compute mass is smaller.
- This supports the stronger claim: interventional value-of-computation is a regime map, not a scalar average.

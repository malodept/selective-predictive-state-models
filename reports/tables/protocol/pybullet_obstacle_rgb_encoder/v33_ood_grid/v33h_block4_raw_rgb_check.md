# v33H block4 raw RGB sanity check

| seed | current RGB max diff | future RGB max diff | future XY max diff | action diff |
|---:|---:|---:|---:|---:|
| 20 | 0 | 164 | 8.34465e-07 | 0 |
| 21 | 0 | 164 | 8.34465e-07 | 0 |
| 22 | 0 | 164 | 8.34465e-07 | 0 |

## Interpretation
- If future RGB differs while future_xy is nearly identical, the equality of metrics is not file reuse; it is a physically similar but visually non-bitwise-identical endpoint.
- If future RGB is identical too, then H=36/H=72 are effectively identical in blocked=4.

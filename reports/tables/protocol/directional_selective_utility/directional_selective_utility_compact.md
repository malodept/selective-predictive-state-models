# Directional selective utility compact summary

Utility is defined as `gain - lambda * selected_fraction`.

| lambda | selected policy | selected utility | always-residual utility | identity utility | selected gain | selected fraction | best test policy among three |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0000 | action_norm router | 0.016816 | 0.017418 | 0.000000 | 0.016816 | 0.947489 | always_residual |
| 0.0025 | action_norm router | 0.014284 | 0.014918 | 0.000000 | 0.016581 | 0.918811 | always_residual |
| 0.0050 | action_norm router | 0.011816 | 0.012418 | 0.000000 | 0.016267 | 0.890192 | always_residual |
| 0.0100 | action_norm router | 0.007235 | 0.007418 | 0.000000 | 0.015078 | 0.784288 | always_residual |
| 0.0150 | action_norm router | 0.003313 | 0.002418 | 0.000000 | 0.015078 | 0.784288 | selected |
| 0.0200 | action_norm router | 0.001713 | -0.002582 | 0.000000 | 0.008167 | 0.322676 | selected |
| 0.0300 | action_norm router | -0.000542 | -0.012582 | 0.000000 | 0.005468 | 0.200353 | identity_only |

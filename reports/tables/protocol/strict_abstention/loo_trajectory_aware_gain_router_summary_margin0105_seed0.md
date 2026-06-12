# Leave-one-trajectory-out trajectory-aware gain-router calibration

Lambda compute: `0.0400`. Marginal expensive threshold: `0.1200`.

The trajectory-aware variants choose routing thresholds from inner held-out trajectories rather than from a random sample-level calibration split.

| method | error | compute | selected | utility | delta utility vs cheap-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| cheap-only | 2.1173 ± 0.1981 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | -2.1573 ± 0.1981 | 0.0000 ± 0.0000 |
| all-expensive | 2.1486 ± 0.2216 | 4.0000 ± 0.0000 | 1.0000 ± 0.0000 | -2.3086 ± 0.2216 | -0.1514 ± 0.0350 |
| sample RF | 2.1198 ± 0.2007 | 1.6298 ± 1.1367 | 0.2099 ± 0.3789 | -2.1850 ± 0.2015 | -0.0278 ± 0.0563 |
| sample RF + fallback | 2.1164 ± 0.1985 | 1.2500 ± 0.8660 | 0.0833 ± 0.2887 | -2.1664 ± 0.1967 | -0.0091 ± 0.0316 |
| trajectory-aware RF median | 2.1208 ± 0.1994 | 1.6537 ± 0.9224 | 0.2179 ± 0.3075 | -2.1869 ± 0.1965 | -0.0297 ± 0.0390 |
| trajectory-aware RF conservative | 2.1188 ± 0.1984 | 1.4374 ± 0.9100 | 0.1458 ± 0.3033 | -2.1763 ± 0.1989 | -0.0190 ± 0.0357 |
| trajectory-aware RF safe fallback | 2.1173 ± 0.1981 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | -2.1573 ± 0.1981 | 0.0000 ± 0.0000 |
| oracle upper bound | 2.0746 ± 0.2115 | 1.4895 ± 0.1945 | 0.1632 ± 0.0648 | -2.1342 ± 0.2059 | 0.0231 ± 0.0114 |

## Inner trajectory calibration folds

Inner calibration folds: `60`.
Mean inner calibration ΔU: `0.0151`.
Minimum inner calibration ΔU: `0.0000`.

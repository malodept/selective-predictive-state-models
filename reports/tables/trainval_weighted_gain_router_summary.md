# Train-to-validation weighted gain-router experiment

Lambda compute: `0.0400`. Marginal expensive threshold: `0.1200`.

| method | error | compute | selected | utility | delta utility vs cheap-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| cheap-only | 1.2568 ± 0.0036 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | -1.2968 ± 0.0036 | 0.0000 ± 0.0000 |
| all-expensive | 1.1409 ± 0.0007 | 4.0000 ± 0.0000 | 1.0000 ± 0.0000 | -1.3009 ± 0.0007 | -0.0042 ± 0.0030 |
| unweighted classifier | 1.1537 ± 0.0034 | 3.0650 ± 0.1246 | 0.6883 ± 0.0415 | -1.2763 ± 0.0026 | 0.0204 ± 0.0016 |
| sqrt-weighted classifier | 1.1559 ± 0.0041 | 3.0769 ± 0.1552 | 0.6923 ± 0.0517 | -1.2790 ± 0.0060 | 0.0178 ± 0.0025 |
| weighted classifier | 1.1558 ± 0.0037 | 3.0926 ± 0.1408 | 0.6975 ± 0.0469 | -1.2795 ± 0.0048 | 0.0172 ± 0.0016 |
| hardband weighted classifier | 1.1556 ± 0.0057 | 3.1160 ± 0.1160 | 0.7053 ± 0.0387 | -1.2803 ± 0.0054 | 0.0165 ± 0.0026 |
| soft utility router | 1.1442 ± 0.0015 | 3.8420 ± 0.1002 | 0.9473 ± 0.0334 | -1.2979 ± 0.0025 | -0.0011 ± 0.0021 |
| oracle upper bound | 1.0823 ± 0.0070 | 2.6965 ± 0.0409 | 0.5655 ± 0.0136 | -1.1901 ± 0.0058 | 0.1067 ± 0.0093 |

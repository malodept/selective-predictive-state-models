# Train-to-validation pairwise ranking gain router

Lambda compute: `0.0400`. Marginal expensive threshold: `0.1200`.

| method | error | compute | selected | utility | delta utility vs cheap-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| cheap-only | 1.2568 ± 0.0036 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | -1.2968 ± 0.0036 | 0.0000 ± 0.0000 |
| all-expensive | 1.1409 ± 0.0007 | 4.0000 ± 0.0000 | 1.0000 ± 0.0000 | -1.3009 ± 0.0007 | -0.0042 ± 0.0030 |
| pairwise ranking router | 1.1553 ± 0.0019 | 3.1018 ± 0.0731 | 0.7006 ± 0.0244 | -1.2794 ± 0.0046 | 0.0174 ± 0.0017 |
| oracle upper bound | 1.0823 ± 0.0070 | 2.6965 ± 0.0409 | 0.5655 ± 0.0136 | -1.1901 ± 0.0058 | 0.1067 ± 0.0093 |

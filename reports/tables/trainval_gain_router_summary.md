# Train-to-validation gain-router experiment, DINOv2 best-validation

Lambda compute: `0.0400`. Marginal expensive threshold: `0.1200`.

| method | error | compute | selected | utility | delta utility vs cheap-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| cheap-only | 1.2568 ± 0.0036 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | -1.2968 ± 0.0036 | 0.0000 ± 0.0000 |
| all-expensive | 1.1409 ± 0.0007 | 4.0000 ± 0.0000 | 1.0000 ± 0.0000 | -1.3009 ± 0.0007 | -0.0042 ± 0.0030 |
| reliability threshold | 1.1411 ± 0.0006 | 3.9915 ± 0.0114 | 0.9972 ± 0.0038 | -1.3007 ± 0.0010 | -0.0040 ± 0.0027 |
| action norm threshold | 1.1437 ± 0.0008 | 3.8329 ± 0.0792 | 0.9443 ± 0.0264 | -1.2971 ± 0.0024 | -0.0003 ± 0.0016 |
| hybrid threshold | 1.1455 ± 0.0049 | 3.8065 ± 0.2208 | 0.9355 ± 0.0736 | -1.2977 ± 0.0041 | -0.0010 ± 0.0006 |
| trainval gain classifier | 1.1528 ± 0.0041 | 3.1464 ± 0.1326 | 0.7155 ± 0.0442 | -1.2787 ± 0.0043 | 0.0181 ± 0.0009 |
| trainval gain regressor | 1.1613 ± 0.0047 | 2.9988 ± 0.2015 | 0.6663 ± 0.0672 | -1.2812 ± 0.0048 | 0.0155 ± 0.0012 |
| oracle upper bound | 1.0823 ± 0.0070 | 2.6965 ± 0.0409 | 0.5655 ± 0.0136 | -1.1901 ± 0.0058 | 0.1067 ± 0.0093 |

# Gain-router baselines, DINOv2 best-validation, lambda=0.040

Marginal compute threshold for using the expensive predictor: `0.1200`.

| method | error | compute | selected | utility |
| --- | ---: | ---: | ---: | ---: |
| cheap-only | 1.2339 ± 0.0300 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | -1.2739 ± 0.0300 |
| all-expensive | 1.1186 ± 0.0297 | 4.0000 ± 0.0000 | 1.0000 ± 0.0000 | -1.2786 ± 0.0297 |
| learned reliability | 1.1875 ± 0.0182 | 1.8919 ± 0.2416 | 0.2973 ± 0.0805 | -1.2632 ± 0.0278 |
| action norm | 1.1754 ± 0.0141 | 2.1361 ± 0.5237 | 0.3787 ± 0.1746 | -1.2608 ± 0.0275 |
| hybrid rel/action alpha=0.25 | 1.1821 ± 0.0234 | 1.9435 ± 0.1715 | 0.3145 ± 0.0572 | -1.2599 ± 0.0277 |
| hybrid rel/action alpha=0.50 | 1.1860 ± 0.0202 | 1.8317 ± 0.4568 | 0.2772 ± 0.1523 | -1.2592 ± 0.0285 |
| hybrid rel/action alpha=0.75 | 1.1752 ± 0.0064 | 2.1422 ± 0.7604 | 0.3807 ± 0.2535 | -1.2608 ± 0.0271 |
| random same fraction | 1.1965 ± 0.0172 | 1.8852 ± 0.2668 | 0.2951 ± 0.0889 | -1.2719 ± 0.0279 |
| learned gain small, theoretical threshold | 1.1753 ± 0.0151 | 2.0802 ± 0.3537 | 0.3601 ± 0.1179 | -1.2585 ± 0.0292 |
| learned gain small, calibrated threshold | 1.1794 ± 0.0061 | 2.0182 ± 0.6340 | 0.3394 ± 0.2113 | -1.2601 ± 0.0286 |
| learned gain state-action, theoretical threshold | 1.1704 ± 0.0224 | 2.4137 ± 0.1346 | 0.4712 ± 0.0449 | -1.2670 ± 0.0275 |
| learned gain state-action, calibrated threshold | 1.1722 ± 0.0207 | 2.3238 ± 0.1866 | 0.4413 ± 0.0622 | -1.2651 ± 0.0267 |
| oracle gain upper bound | 1.0572 ± 0.0191 | 2.7345 ± 0.0398 | 0.5782 ± 0.0133 | -1.1666 ± 0.0201 |

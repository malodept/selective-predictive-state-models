# Leave-one-trajectory-out learned gain router, DINOv2, seed 0

Lambda compute: `0.0400`. Marginal expensive threshold: `0.1200`.
Router: random forest classifier trained on non-held-out trajectory data, calibrated on `0.25` of training samples.

| heldout | run | method | error | compute | selected | utility | ΔU vs cheap | calib ΔU | threshold |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P000 | cheap10_exp20 | cheap-only | 1.8228 | 1.0000 | 0.0000 | -1.8628 | 0.0000 |  |  |
| P000 | cheap10_exp20 | all-expensive | 1.8636 | 4.0000 | 1.0000 | -2.0236 | -0.1608 |  |  |
| P000 | cheap10_exp20 | learned RF gain router | 1.8228 | 1.0000 | 0.0000 | -1.8628 | 0.0000 | 0.0000 | 0.9520 |
| P000 | cheap10_exp20 | learned RF gain router + fallback | 1.8228 | 1.0000 | 0.0000 | -1.8628 | 0.0000 | 0.0000 | 0.9520 |
| P000 | cheap10_exp20 | oracle upper bound | 1.7755 | 1.4843 | 0.1614 | -1.8349 | 0.0279 |  |  |
| P000 | cheap5_exp20 | cheap-only | 1.8506 | 1.0000 | 0.0000 | -1.8906 | 0.0000 |  |  |
| P000 | cheap5_exp20 | all-expensive | 1.8367 | 4.0000 | 1.0000 | -1.9967 | -0.1061 |  |  |
| P000 | cheap5_exp20 | learned RF gain router | 1.8468 | 1.7051 | 0.2350 | -1.9150 | -0.0243 | 0.0067 | 0.6509 |
| P000 | cheap5_exp20 | learned RF gain router + fallback | 1.8468 | 1.7051 | 0.2350 | -1.9150 | -0.0243 | 0.0067 | 0.6509 |
| P000 | cheap5_exp20 | oracle upper bound | 1.7618 | 1.9244 | 0.3081 | -1.8388 | 0.0518 |  |  |
| P001 | cheap10_exp20 | cheap-only | 2.4766 | 1.0000 | 0.0000 | -2.5166 | 0.0000 |  |  |
| P001 | cheap10_exp20 | all-expensive | 2.5433 | 4.0000 | 1.0000 | -2.7033 | -0.1868 |  |  |
| P001 | cheap10_exp20 | learned RF gain router | 2.4766 | 1.1899 | 0.0633 | -2.5242 | -0.0076 | 0.0064 | 0.6939 |
| P001 | cheap10_exp20 | learned RF gain router + fallback | 2.4766 | 1.1899 | 0.0633 | -2.5242 | -0.0076 | 0.0064 | 0.6939 |
| P001 | cheap10_exp20 | oracle upper bound | 2.4506 | 1.2925 | 0.0975 | -2.5023 | 0.0142 |  |  |
| P001 | cheap5_exp20 | cheap-only | 2.4766 | 1.0000 | 0.0000 | -2.5166 | 0.0000 |  |  |
| P001 | cheap5_exp20 | all-expensive | 2.5489 | 4.0000 | 1.0000 | -2.7089 | -0.1923 |  |  |
| P001 | cheap5_exp20 | learned RF gain router | 2.4766 | 1.0000 | 0.0000 | -2.5166 | 0.0000 | 0.0000 | 0.9348 |
| P001 | cheap5_exp20 | learned RF gain router + fallback | 2.4766 | 1.0000 | 0.0000 | -2.5166 | 0.0000 | 0.0000 | 0.9348 |
| P001 | cheap5_exp20 | oracle upper bound | 2.4554 | 1.2804 | 0.0935 | -2.5066 | 0.0099 |  |  |
| P002 | cheap10_exp20 | cheap-only | 2.1365 | 1.0000 | 0.0000 | -2.1765 | 0.0000 |  |  |
| P002 | cheap10_exp20 | all-expensive | 2.1937 | 4.0000 | 1.0000 | -2.3537 | -0.1772 |  |  |
| P002 | cheap10_exp20 | learned RF gain router | 2.1365 | 1.0000 | 0.0000 | -2.1765 | 0.0000 | 0.0001 | 0.9836 |
| P002 | cheap10_exp20 | learned RF gain router + fallback | 2.1365 | 1.0000 | 0.0000 | -2.1765 | 0.0000 | 0.0001 | 0.9836 |
| P002 | cheap10_exp20 | oracle upper bound | 2.0922 | 1.4856 | 0.1619 | -2.1516 | 0.0249 |  |  |
| P002 | cheap5_exp20 | cheap-only | 2.1383 | 1.0000 | 0.0000 | -2.1783 | 0.0000 |  |  |
| P002 | cheap5_exp20 | all-expensive | 2.1949 | 4.0000 | 1.0000 | -2.3549 | -0.1767 |  |  |
| P002 | cheap5_exp20 | learned RF gain router | 2.1949 | 4.0000 | 1.0000 | -2.3549 | -0.1767 | 0.0521 | 0.3564 |
| P002 | cheap5_exp20 | learned RF gain router + fallback | 2.1949 | 4.0000 | 1.0000 | -2.3549 | -0.1767 | 0.0521 | 0.3564 |
| P002 | cheap5_exp20 | oracle upper bound | 2.1029 | 1.4902 | 0.1634 | -2.1625 | 0.0158 |  |  |
| P003 | cheap10_exp20 | cheap-only | 2.0900 | 1.0000 | 0.0000 | -2.1300 | 0.0000 |  |  |
| P003 | cheap10_exp20 | all-expensive | 2.1037 | 4.0000 | 1.0000 | -2.2637 | -0.1337 |  |  |
| P003 | cheap10_exp20 | learned RF gain router | 2.0900 | 1.0000 | 0.0000 | -2.1300 | 0.0000 | 0.0000 | 0.8943 |
| P003 | cheap10_exp20 | learned RF gain router + fallback | 2.0900 | 1.0000 | 0.0000 | -2.1300 | 0.0000 | 0.0000 | 0.8943 |
| P003 | cheap10_exp20 | oracle upper bound | 2.0533 | 1.4882 | 0.1627 | -2.1128 | 0.0172 |  |  |
| P003 | cheap5_exp20 | cheap-only | 2.0927 | 1.0000 | 0.0000 | -2.1327 | 0.0000 |  |  |
| P003 | cheap5_exp20 | all-expensive | 2.1208 | 4.0000 | 1.0000 | -2.2808 | -0.1481 |  |  |
| P003 | cheap5_exp20 | learned RF gain router | 2.0927 | 1.0000 | 0.0000 | -2.1327 | 0.0000 | 0.0000 | 0.9905 |
| P003 | cheap5_exp20 | learned RF gain router + fallback | 2.0927 | 1.0000 | 0.0000 | -2.1327 | 0.0000 | 0.0000 | 0.9905 |
| P003 | cheap5_exp20 | oracle upper bound | 2.0673 | 1.3451 | 0.1150 | -2.1211 | 0.0116 |  |  |
| P004 | cheap10_exp20 | cheap-only | 2.1259 | 1.0000 | 0.0000 | -2.1659 | 0.0000 |  |  |
| P004 | cheap10_exp20 | all-expensive | 2.1727 | 4.0000 | 1.0000 | -2.3327 | -0.1668 |  |  |
| P004 | cheap10_exp20 | learned RF gain router | 2.1259 | 1.0000 | 0.0000 | -2.1659 | 0.0000 | 0.0000 | 0.9099 |
| P004 | cheap10_exp20 | learned RF gain router + fallback | 2.1259 | 1.0000 | 0.0000 | -2.1659 | 0.0000 | 0.0000 | 0.9099 |
| P004 | cheap10_exp20 | oracle upper bound | 2.0779 | 1.3827 | 0.1276 | -2.1332 | 0.0328 |  |  |
| P004 | cheap5_exp20 | cheap-only | 2.1259 | 1.0000 | 0.0000 | -2.1659 | 0.0000 |  |  |
| P004 | cheap5_exp20 | all-expensive | 2.1787 | 4.0000 | 1.0000 | -2.3387 | -0.1728 |  |  |
| P004 | cheap5_exp20 | learned RF gain router | 2.1259 | 1.0000 | 0.0000 | -2.1659 | 0.0000 | 0.0000 | 0.9697 |
| P004 | cheap5_exp20 | learned RF gain router + fallback | 2.1259 | 1.0000 | 0.0000 | -2.1659 | 0.0000 | 0.0000 | 0.9697 |
| P004 | cheap5_exp20 | oracle upper bound | 2.0891 | 1.3192 | 0.1064 | -2.1419 | 0.0241 |  |  |
| P005 | cheap10_exp20 | cheap-only | 2.0330 | 1.0000 | 0.0000 | -2.0730 | 0.0000 |  |  |
| P005 | cheap10_exp20 | all-expensive | 1.9987 | 4.0000 | 1.0000 | -2.1587 | -0.0857 |  |  |
| P005 | cheap10_exp20 | learned RF gain router | 2.0214 | 1.6620 | 0.2207 | -2.0878 | -0.0149 | 0.0159 | 0.5776 |
| P005 | cheap10_exp20 | learned RF gain router + fallback | 2.0214 | 1.6620 | 0.2207 | -2.0878 | -0.0149 | 0.0159 | 0.5776 |
| P005 | cheap10_exp20 | oracle upper bound | 1.9777 | 1.7231 | 0.2410 | -2.0466 | 0.0264 |  |  |
| P005 | cheap5_exp20 | cheap-only | 2.0383 | 1.0000 | 0.0000 | -2.0783 | 0.0000 |  |  |
| P005 | cheap5_exp20 | all-expensive | 2.0279 | 4.0000 | 1.0000 | -2.1879 | -0.1096 |  |  |
| P005 | cheap5_exp20 | learned RF gain router | 2.0279 | 4.0000 | 1.0000 | -2.1879 | -0.1096 | 0.1187 | 0.4369 |
| P005 | cheap5_exp20 | learned RF gain router + fallback | 2.0279 | 4.0000 | 1.0000 | -2.1879 | -0.1096 | 0.1187 | 0.4369 |
| P005 | cheap5_exp20 | oracle upper bound | 1.9914 | 1.6590 | 0.2197 | -2.0578 | 0.0205 |  |  |

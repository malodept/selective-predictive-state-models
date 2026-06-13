# Trajectory-level bootstrap for cheap_residual

Metric: per-trajectory mean of `identity_error - calibrated_residual_error` on the held-out test environment.

Seed-trajectory units: `54`.
Trajectory units after seed averaging: `18`.

| aggregation | mean improvement | 95% bootstrap CI low | 95% bootstrap CI high |
| --- | ---: | ---: | ---: |
| seed-trajectory equal weight | 0.007730 | 0.007142 | 0.008289 |
| trajectory equal weight | 0.007730 | 0.006805 | 0.008663 |

## Per-trajectory effects averaged over seeds

| trajectory | samples | identity error | model error | improvement vs identity | ratio vs identity |
| --- | ---: | ---: | ---: | ---: | ---: |
| neighborhood/Hard/P001 | 2985 | 1.768190 | 1.757220 | 0.010970 | 0.993796 |
| neighborhood/Hard/P014 | 2755 | 1.820639 | 1.810021 | 0.010618 | 0.994168 |
| neighborhood/Hard/P015 | 2985 | 1.793858 | 1.783242 | 0.010616 | 0.994082 |
| neighborhood/Hard/P003 | 2920 | 1.617841 | 1.608591 | 0.009249 | 0.994283 |
| neighborhood/Hard/P013 | 2470 | 1.687144 | 1.677918 | 0.009226 | 0.994532 |
| neighborhood/Hard/P009 | 1490 | 1.771690 | 1.762636 | 0.009054 | 0.994889 |
| neighborhood/Hard/P005 | 2985 | 1.778210 | 1.769416 | 0.008794 | 0.995055 |
| neighborhood/Hard/P007 | 2650 | 1.444736 | 1.436588 | 0.008148 | 0.994360 |
| neighborhood/Hard/P002 | 2985 | 1.698298 | 1.690577 | 0.007721 | 0.995454 |
| neighborhood/Hard/P006 | 2875 | 1.725839 | 1.718326 | 0.007513 | 0.995647 |
| neighborhood/Hard/P012 | 2985 | 1.520765 | 1.513344 | 0.007421 | 0.995120 |
| neighborhood/Hard/P000 | 2985 | 1.383246 | 1.376085 | 0.007161 | 0.994823 |
| neighborhood/Hard/P011 | 2985 | 1.603093 | 1.596013 | 0.007080 | 0.995583 |
| neighborhood/Hard/P004 | 2985 | 1.636128 | 1.630117 | 0.006011 | 0.996326 |
| neighborhood/Hard/P010 | 2985 | 1.345032 | 1.339677 | 0.005355 | 0.996019 |
| neighborhood/Hard/P008 | 2985 | 1.456221 | 1.450968 | 0.005253 | 0.996393 |
| neighborhood/Hard/P016 | 2985 | 1.575139 | 1.570192 | 0.004947 | 0.996860 |
| neighborhood/Hard/P017 | 2985 | 1.398708 | 1.394698 | 0.004010 | 0.997133 |

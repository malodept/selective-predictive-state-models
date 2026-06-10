# Routing signal diagnostics, DINOv2 best-validation

Marginal expensive-compute threshold: `0.1200`.

| seed | n | mean true gain | profitable fraction | oracle ΔU | all-exp ΔU | reliability AUROC | action AUROC | hybrid AUROC | reliability top-k ΔU | action top-k ΔU | hybrid top-k ΔU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 3292 | 0.1117 | 0.5465 | 0.0951 | -0.0083 | 0.5740 | 0.6266 | 0.6293 | 0.0064 | 0.0125 | 0.0130 |
| 1 | 3292 | 0.1187 | 0.5723 | 0.1178 | -0.0013 | 0.5951 | 0.6173 | 0.6324 | 0.0107 | 0.0157 | 0.0170 |
| 2 | 3292 | 0.1171 | 0.5778 | 0.1070 | -0.0029 | 0.5703 | 0.5965 | 0.6023 | 0.0058 | 0.0165 | 0.0118 |
| mean | 9876 | 0.1158 | 0.5655 | 0.1067 | -0.0042 | 0.5798 | 0.6135 | 0.6213 | 0.0076 | 0.0149 | 0.0140 |
| std |  | 0.0030 | 0.0136 | 0.0093 | 0.0030 | 0.0109 | 0.0126 | 0.0135 | 0.0022 | 0.0017 | 0.0022 |

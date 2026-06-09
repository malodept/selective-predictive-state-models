# DINOv2 error-supervised reliability multi-seed summary

| metric | mean | std | values |
| --- | ---: | ---: | --- |
| R@1 | 0.1405 | 0.0015 | 0.1424, 0.1388, 0.1403 |
| R@5 | 0.5753 | 0.0049 | 0.5806, 0.5689, 0.5764 |
| R@10 | 0.8304 | 0.0069 | 0.8402, 0.8256, 0.8256 |
| expected_learned_auroc | 0.7634 | 0.0026 | 0.7609, 0.7623, 0.7669 |
| expected_residual_auroc | 0.9795 | 0.0000 | 0.9795, 0.9794, 0.9795 |
| observed_residual_auroc | 0.9841 | 0.0007 | 0.9849, 0.9832, 0.9841 |
| observed_learned_auroc | 0.4958 | 0.0027 | 0.4993, 0.4927, 0.4952 |

## Best selector policy per seed, fixed lambda from config

| seed | policy | error | compute | selected | utility |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | all-expensive | 0.5037 | 4.0000 | 1.0000 | -0.6637 |
| 1 | all-expensive | 0.5115 | 4.0000 | 1.0000 | -0.6715 |
| 2 | all-expensive | 0.5102 | 4.0000 | 1.0000 | -0.6702 |

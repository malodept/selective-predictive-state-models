| run | R@1 | R@5 | R@10 | expected_learned_AUROC | expected_residual_AUROC | observed_residual_AUROC | best_policy | best_error | best_compute | best_selected | best_utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 heuristic | 0.0790 | 0.3635 | 0.6105 | 0.9881 | 0.5790 | 0.9602 | cheap-only | 0.2107 | 1.0000 | 0.0000 | -0.4607 |
| ResNet18 error-rel | 0.0673 | 0.3151 | 0.5450 | 0.7602 | 0.9797 | 0.9554 | threshold=0.50 | 0.1951 | 1.5714 | 0.1905 | -0.2579 |
| DINOv2 heuristic | 0.1363 | 0.5860 | 0.8394 | 0.9779 | 0.5506 | 0.9845 | threshold=0.20 | 0.5482 | 3.4550 | 0.8183 | -0.6864 |
| DINOv2 error-rel | 0.1378 | 0.5760 | 0.8334 | 0.7669 | 0.9780 | 0.9845 | all-expensive | 0.5156 | 4.0000 | 1.0000 | -0.6756 |
| DINOv2 L2 heuristic | 0.0000 | 0.0011 | 0.0032 | 0.9912 | 0.3683 | 0.4912 | cheap-only | 0.0057 | 1.0000 | 0.0000 | -0.0457 |
| DINOv2 L2 error-rel | 0.0018 | 0.0064 | 0.0110 | 0.7636 | 0.4926 | 0.5209 | cheap-only | 0.0036 | 1.0000 | 0.0000 | -0.0436 |

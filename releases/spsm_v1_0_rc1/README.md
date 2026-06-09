# SPSM v1.0-rc1

This release candidate documents the current state of the Selective Predictive-State Models project.

## Core idea

The system learns to predict future latent observations and to estimate when its own cheap prediction is likely to fail. The predicted unreliability score can then be used to decide whether optional extra computation should be activated.

## Dataset

- Dataset: TartanAir JapaneseAlley-Hard
- Trajectories: P000 to P005
- Transitions: 13,170
- Input: RGB frames and camera poses
- Action representation: pose difference between frame t and frame t+k

## Encoders

Two frozen latent teachers are evaluated:

- ResNet18 ImageNet
- DINOv2 ViT-S/14

## Main result

The key result is that reliability becomes much more meaningful when supervised by the actual future error of the cheap predictor rather than by a hand-crafted motion heuristic.

Expected residual AUROC:

| Encoder | Heuristic target | Error-supervised target |
|---|---:|---:|
| ResNet18 | 0.579 | 0.980 |
| DINOv2 | 0.551 | 0.978 |

This means that error-supervised reliability makes the expected difficulty label much more aligned with realized prediction error.

## Retrieval

DINOv2 substantially improves future latent retrieval:

| Encoder | R@10 |
|---|---:|
| ResNet18 heuristic | 0.611 |
| DINOv2 heuristic | 0.839 |
| DINOv2 error-supervised | 0.833 |

## Selective compute

For ResNet18 error-supervised reliability, the adaptive selector improves utility over both cheap-only and all-expensive at the selected compute penalty.

For DINOv2, the prediction space is stronger, but the all-expensive policy remains optimal at lambda = 0.04. This indicates that the compute-cost regime depends on the latent teacher and should be evaluated through full utility-vs-compute curves.

## Files

- `figures/`: main figures for the report
- `tables/`: markdown tables with main metrics
- `configs/`: training configs used for the main runs
- `scripts/`: reproduction helpers

## Status

This is a controlled benchmark release, not yet a final paper-ready system. The next technical steps are:

1. replace the refinement proxy with a real expensive predictor;
2. evaluate utility-vs-compute curves across lambda values;
3. test cosine-compatible DINOv2 training;
4. extend to video-level teachers such as V-JEPA;
5. evaluate multi-step rollout prediction.

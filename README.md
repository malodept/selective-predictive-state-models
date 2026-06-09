# Selective Predictive-State Models

Reliability-aware latent world models for selective computation.

This repository contains an experimental implementation of **Selective Predictive-State Models (SPSM)**. The goal is to learn a latent predictive model that not only predicts future observations, but also estimates when its own cheap prediction is likely to fail. This reliability estimate can then be used to decide whether additional computation should be activated.

The current benchmark uses TartanAir visual trajectories, frozen visual encoders, pose-conditioned latent prediction, and reliability-driven selective refinement.

---

## Core idea

Given a current observation \(x_t\), a future observation \(x_{t+k}\), and an action or motion descriptor \(a_{t,k}\), a frozen encoder maps images to latent states:

\[
z_t = E(x_t), \qquad z_{t+k} = E(x_{t+k})
\]

A cheap predictive model estimates the future latent:

\[
\hat z_{t+k} = f_{\text{cheap}}(z_t, a_{t,k})
\]

A reliability head estimates whether this cheap prediction is likely to be unreliable:

\[
r_{t,k} = R_\theta(z_t, a_{t,k}, \hat z_{t+k})
\]

If \(r_{t,k}\) is high, the system can activate optional extra computation.

---

## Why this matters

A standard predictive world model asks:

> What will I observe next?

SPSM asks an additional question:

> Can I know in advance when my own prediction is likely to be wrong?

This distinction is important for agents that must operate under limited compute. Instead of always running an expensive predictor, the system can use a learned reliability signal to allocate computation selectively.

---

## Dataset

Current experiments use:

- **TartanAir JapaneseAlley-Hard**
- 6 trajectories: `P000` to `P005`
- 13,170 transitions
- RGB frames and camera poses
- action representation: camera-pose difference between frame \(t\) and frame \(t+k\)

The action is not a motor command in the current benchmark. It is a pose-difference descriptor used to condition latent transition prediction.

---

## Encoders

Two frozen latent teachers are evaluated:

| Encoder | Latent dimension |
|---|---:|
| ResNet18 ImageNet | 512 |
| DINOv2 ViT-S/14 | 384 |

The encoders are frozen. The project focuses on the predictive-state and reliability mechanism, not on fine-tuning visual representations.

---

## Reliability targets

Two reliability targets are compared.

### 1. Heuristic reliability

A transition is labeled difficult if the temporal gap or pose displacement is large.

This is easy to learn, but it does not necessarily correspond to actual prediction error.

### 2. Error-supervised reliability

A cheap predictor is trained first. Then transitions are labeled difficult if the cheap predictor has high realized error on them.

This asks the reliability head to predict:

> Will my cheap predictor likely fail on this transition?

This is the main reliability formulation used in the current results.

---

## Main results

### Error-supervised reliability aligns difficulty with real prediction error

Expected residual AUROC measures whether the difficulty label is aligned with actual realized prediction error.

| Encoder | Heuristic target | Error-supervised target |
|---|---:|---:|
| ResNet18 | 0.579 | 0.980 |
| DINOv2 | 0.551 | 0.978 |

This is the central result: hand-crafted difficulty labels are easy to learn but poorly aligned with actual model errors, while error-supervised reliability is much more meaningful.

### DINOv2 improves latent future retrieval

| Encoder / target | R@1 | R@5 | R@10 |
|---|---:|---:|---:|
| ResNet18 heuristic | 0.079 | 0.364 | 0.611 |
| ResNet18 error-supervised | 0.067 | 0.315 | 0.545 |
| DINOv2 heuristic | 0.136 | 0.586 | 0.839 |
| DINOv2 error-supervised | 0.138 | 0.576 | 0.833 |

DINOv2 provides a stronger latent space for future-state retrieval.

### Selective compute

For ResNet18 error-supervised reliability, the learned reliability score enables an adaptive policy that improves utility over both cheap-only and all-expensive execution at the selected compute penalty.

For DINOv2, all-expensive remains optimal at the default compute penalty, indicating that the optimal compute regime depends on the latent teacher and should be evaluated using full utility-vs-compute curves.

---

## Key metrics

The four reliability and surprise metrics are:

| Metric | Score used | Target | Meaning |
|---|---|---|---|
| Expected learned AUROC | predicted unreliability | expected difficulty | Can the model predict before observation which transitions are hard? |
| Expected residual AUROC | realized residual error | expected difficulty | Are difficult transitions actually high-error transitions? |
| Observed learned AUROC | predicted unreliability | observed surprise | Can the model predict unpredictable future mismatches? |
| Observed residual AUROC | realized residual error | observed surprise | Can residual error detect surprises after observation? |

Observed learned AUROC is expected to be near chance when the surprise is injected after the fact. Observed residual AUROC should be high if the prediction residual detects mismatched futures.

---

## Repository structure

```text
configs/        Training configurations
scripts/        Data preparation, feature extraction, plotting and reporting scripts
src/spsm/       Core Python package
tests/          Unit tests
outputs/        Experiment outputs, ignored in public releases if too large
reports/        Figures, tables and report files
releases/       Release snapshots
Reproducing the current benchmark

Run the unit tests:

apptainer exec /shared/projects/phisat2/containers/phisat2.sif \
  python -m pytest tests

Regenerate the main ablation table:

apptainer exec /shared/projects/phisat2/containers/phisat2.sif \
  python scripts/teacher_ablation_main_table.py

The current release helper is:

bash releases/spsm_v1_0_rc1/scripts/reproduce_main_runs.sh
Current status

This repository is currently at v1.0-rc1.

Validated:

ResNet18 error-supervised reliability across 3 seeds
DINOv2 error-supervised reliability across 3 seeds
teacher ablation table
reliability and surprise diagnostics
utility-vs-compute plots
unit tests

Not yet implemented:

real trained expensive refinement predictor
cosine-compatible DINOv2 objective
video-level teachers such as V-JEPA
multi-step rollout evaluation
real robot or driving transfer
Roadmap

Next technical steps:

replace the refinement proxy with a real expensive predictor;
evaluate full utility-vs-compute curves across compute penalties;
implement a cosine-compatible DINOv2 variant;
test V-JEPA or another video-level teacher;
extend from single-step latent prediction to multi-step rollouts;
evaluate transfer beyond TartanAir.
Citation

No formal citation yet. This is an early research prototype.

---

## v1.1: Real selective refinement

The `v1.1-real-refinement` branch adds a real selective-refinement experiment.

Instead of using only an evaluation-time refinement proxy, this experiment trains two actual predictors:

- a cheap predictor: small MLP, 10 epochs;
- an expensive predictor: larger MLP, 80 epochs;
- a reliability head: trained to predict high-error transitions of the cheap predictor.

On DINOv2 latents, the adaptive reliability-based selector improves utility over both static baselines across three seeds.

| Policy | Mean utility |
|---|---:|
| cheap-only | -1.4121 |
| all-expensive | -1.4109 |
| adaptive selector | **-1.3932** |

The adaptive selector uses only about 17% expensive executions on average:

| Metric | Mean | Std |
|---|---:|---:|
| best error | 1.3327 | 0.0078 |
| best compute | 1.5140 | 0.0236 |
| selected fraction | 0.1713 | 0.0079 |
| best utility | -1.3932 | 0.0068 |

This result shows that the reliability signal can control a real trained expensive predictor, not only a proxy refinement mechanism.

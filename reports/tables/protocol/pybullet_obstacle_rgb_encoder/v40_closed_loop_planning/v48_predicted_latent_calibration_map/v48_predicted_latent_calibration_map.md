# SPSM v48 Predicted-Latent Calibration Map

This analysis tests whether predicted future latents are physically calibrated and actionable, rather than only discriminative in latent ranking space.

## Physical calibration map

| model | true-latent probe | pred + true probe | pred + model-specific probe | calibration gap | error ratio |
|---|---:|---:|---:|---:|---:|
| Tiny | 0.036 | 2.859 | 0.031 | 2.828 | 92.8x |
| Medium | 0.036 | 2.553 | 0.027 | 2.526 | 94.3x |
| Full | 0.036 | 2.006 | 0.026 | 1.980 | 77.7x |

## One-step decision calibration

| model | uncal acc | cal acc | acc gain | uncal regret | cal regret | regret reduction |
|---|---:|---:|---:|---:|---:|---:|
| Full | 0.398 | 0.781 | 0.383 | 0.247 | 0.041 | 0.207 |
| Tiny | 0.305 | 0.805 | 0.500 | 0.299 | 0.040 | 0.259 |
| Medium | 0.305 | 0.734 | 0.430 | 0.315 | 0.060 | 0.255 |

## Calibrated one-step block sweep vs greedy-vector

| blocked | model | action acc | regret | acc gap vs greedy | regret reduction vs greedy |
|---:|---|---:|---:|---:|---:|
| 1 | Tiny | 0.823 | 0.034 | -0.141 | -0.033 |
| 1 | Full | 0.776 | 0.054 | -0.188 | -0.053 |
| 1 | Medium | 0.714 | 0.075 | -0.250 | -0.074 |
| 2 | Tiny | 0.833 | 0.021 | -0.130 | -0.019 |
| 2 | Full | 0.797 | 0.039 | -0.167 | -0.037 |
| 2 | Medium | 0.740 | 0.050 | -0.224 | -0.048 |
| 3 | Tiny | 0.839 | 0.013 | -0.141 | -0.012 |
| 3 | Full | 0.818 | 0.023 | -0.161 | -0.022 |
| 3 | Medium | 0.760 | 0.041 | -0.219 | -0.040 |
| 4 | Tiny | 0.802 | 0.011 | -0.198 | -0.011 |
| 4 | Full | 0.797 | 0.013 | -0.203 | -0.013 |
| 4 | Medium | 0.745 | 0.020 | -0.255 | -0.020 |

## Obstacle-sensitive challenge-goal summary

| model | mean regret reduction | min regret reduction | mean acc gain | stable blocks |
|---|---:|---:|---:|---:|
| Full | 0.073 | 0.066 | 0.640 | 3/3 |
| Medium | 0.065 | 0.054 | 0.577 | 3/3 |
| Tiny | 0.073 | 0.062 | 0.625 | 3/3 |

## Interpretation

- Predicted latents are not directly physically calibrated: true-future probes fail badly on predicted futures.
- Model-specific probes recover accurate xy predictions, so the physical information is present but represented differently.
- Calibration improves one-step decisions and enables obstacle-sensitive challenge-goal gains.
- Therefore latent ranking, physical calibration, and decision utility are distinct dimensions of predictive usefulness.

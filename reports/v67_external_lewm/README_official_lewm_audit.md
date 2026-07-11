# External Official LeWorldModel SPSM Audit

## Setup

- Model: official LeWorldModel PushT checkpoint converted to local object checkpoint.
- Environment: `swm/PushT-v1`.
- Candidate construction: weak-policy-style candidate actions.
- Action format: native PushT action is 2D, LeWM action chunk is `frameskip=5`, therefore model action dimension is 10.
- Main setting: `dist_constraint=200`, `K=5`, `n=20000` states per seed, 6 seeds.

## Main result at dist=200

| system | top-1 | MRR | mean rank |
|---|---:|---:|---:|
| clean action | 0.298 ± 0.002 | 0.539 ± 0.002 | 2.59 ± 0.01 |
| zero action | 0.234 ± 0.002 | 0.481 ± 0.001 | 2.90 ± 0.01 |
| random action | 0.234 ± 0.001 | 0.481 ± 0.001 | 2.90 ± 0.01 |
| negative action | 0.197 ± 0.001 | 0.445 ± 0.001 | 3.10 ± 0.01 |
| shuffled prediction | 0.227 ± 0.003 | 0.477 ± 0.002 | 2.91 ± 0.01 |

## Interpretation

The official LeWorldModel checkpoint shows a stable action-conditioned predictive signal under SPSM. Clean action-conditioned predictions outperform zero-action, random-action, negative-action, and shuffled-prediction controls across 6 seeds. This supports the use of SPSM as an external audit protocol rather than only an evaluation of our own predictors.

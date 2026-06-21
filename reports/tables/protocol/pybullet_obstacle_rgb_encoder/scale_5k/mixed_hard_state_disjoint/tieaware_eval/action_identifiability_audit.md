# Action identifiability audit

| action | test groups | same-action pairs | exact delta ties | exact delta tie frac | mean delta distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| stay | 181 | 362 | 362 | 1.000000 | 0.00000000 |
| right | 192 | 384 | 0 | 0.000000 | 0.12145826 |
| left | 205 | 410 | 0 | 0.000000 | 0.13160072 |
| forward | 200 | 400 | 0 | 0.000000 | 0.14116240 |
| backward | 222 | 444 | 0 | 0.000000 | 0.15987233 |

## Interpretation

This audit checks whether same-action/different-state negatives are actually identifiable in latent-displacement space.
If the `stay` action produces exact zero deltas for multiple states, those negatives are not distinguishable by any delta-prediction model.

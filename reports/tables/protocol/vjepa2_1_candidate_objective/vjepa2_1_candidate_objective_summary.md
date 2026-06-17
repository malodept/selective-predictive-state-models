# V-JEPA 2.1 candidate objective summary

This experiment trains the V-JEPA 2.1 patch-token Action Transformer with an additional diagonal candidate-matching objective.

Loss: `MSE + 0.2 * candidate_matching_CE`, temperature `0.1`.

## Prediction metrics

| model | test global error | test gain | test cosine | positive cosine frac |
| --- | ---: | ---: | ---: | ---: |
| V-JEPA 2.1 MSE | 0.027311 | 0.003826 | 0.310250 | 0.998725 |
| V-JEPA 2.1 candidate λ=0.2 | 0.028214 | 0.002923 | 0.264019 | 0.993429 |

The candidate objective worsens global prediction quality relative to the MSE baseline.

## Candidate matching at calibrated alpha

| model | alpha | original top-1 | zero top-1 | shuffled top-1 |
| --- | ---: | ---: | ---: | ---: |
| V-JEPA 2.1 candidate λ=0.2 | 0.505 | 0.200325 | 0.200000 | 0.199925 |

At the MSE-calibrated alpha, candidate matching remains at chance.

## Candidate matching at alpha=1

| split | original top-1 | zero top-1 | shuffled top-1 |
| --- | ---: | ---: | ---: |
| train | 0.230825 | 0.200000 | 0.200675 |
| test | 0.205775 | 0.200000 | 0.200200 |

The candidate objective creates some action-conditioned signal, but it remains weak and does not become robust OOD future discrimination.

## Action sensitivity

| model | prediction spread | true future spread | spread ratio |
| --- | ---: | ---: | ---: |
| V-JEPA 2.1 MSE | 0.004249 | 0.019682 | 0.240714 |
| V-JEPA 2.1 candidate λ=0.2 | 0.007453 | 0.019682 | 0.418139 |

The candidate objective substantially increases prediction sensitivity to the action, but this increased sensitivity is not aligned well enough with the correct future.

## Interpretation

The candidate objective does not solve OOD action-grounding. It increases action-conditioned variation, but the model still fails to rank the matching future above counterfactual futures in the held-out environment.

This suggests that the limitation is not merely insufficient action sensitivity. The core issue is that the current TartanAir protocol does not provide strong enough counterfactual supervision for robust action-grounded future discrimination.

The candidate-objective branch should therefore be reported as a negative but informative result: stronger action sensitivity is possible, but action sensitivity alone is insufficient.

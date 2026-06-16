# V-JEPA 2.1 action-grounding summary

This experiment replaces DINOv2 image patch tokens with frozen V-JEPA 2.1 video tokens while keeping the same TartanAir split, action inputs, patch-token Transformer, and action-grounding diagnostics.

## Main prediction result

| encoder | split | identity error | global error | gain vs identity | relative gain | cosine mean | positive cosine frac |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V-JEPA 2.1 Base causal16 | test | 0.031137 | 0.027311 | 0.003826 | 12.29% | 0.310250 | 0.998725 |

## Action intervention

| inference action | recalibrated gain | recalibrated error | cosine |
| --- | ---: | ---: | ---: |
| original | 0.003826 | 0.027311 | 0.310250 |
| zero | 0.003239 | 0.027898 | 0.306749 |
| shuffled | 0.002635 | 0.028503 | 0.296646 |

Unlike the DINOv2 MSE baseline, changing the action has a visible effect on prediction quality.

## Action sensitivity

| encoder/model | prediction spread | true future spread | spread ratio |
| --- | ---: | ---: | ---: |
| V-JEPA 2.1 MSE | 0.004249 | 0.019682 | 0.240714 |

V-JEPA 2.1 produces much stronger action sensitivity than previous DINOv2-based models.

## Candidate matching

At the calibrated alpha, candidate matching remains near chance.

| split | alpha | original top-1 | zero top-1 | shuffled top-1 |
| --- | ---: | ---: | ---: | ---: |
| train | 0.58 | 0.201325 | 0.200000 | 0.200025 |
| test | 0.58 | 0.200325 | 0.200000 | 0.200000 |

At alpha=1, the in-distribution action signal becomes visible.

| split | alpha | original top-1 | zero top-1 | shuffled top-1 |
| --- | ---: | ---: | ---: | ---: |
| train | 1.00 | 0.250375 | 0.200000 | 0.200875 |
| test | 1.00 | 0.203975 | 0.200000 | 0.200175 |

## Interpretation

V-JEPA 2.1 substantially improves latent temporal predictability and greatly increases action sensitivity compared with DINOv2. It also learns a clear in-distribution action-conditioned signal when evaluated without MSE shrinkage.

However, this action-grounding signal does not transfer robustly to the held-out environment. Therefore, replacing DINOv2 with a video encoder helps, but the current TartanAir protocol still does not provide robust OOD action-grounded prediction.

The next required ablation is a V-JEPA no-action training run. If no-action remains comparable, the protocol is still mostly measuring latent video flow. If no-action drops clearly, then V-JEPA has finally made action useful in the learned dynamics.

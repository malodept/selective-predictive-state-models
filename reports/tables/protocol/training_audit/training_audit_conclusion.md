# Training audit conclusion

This audit investigates why V-JEPA 2.1 dynamics training consistently selected epoch 1 as the best checkpoint under the held-out environment protocol.

## 1. Overfit-batch sanity check

The model can overfit a fixed batch of 64 samples:

| initial loss | final loss | final / initial |
| ---: | ---: | ---: |
| 1.009490 | 0.000352 | 0.000349 |

This rules out a fundamental implementation issue in the target construction, normalization, optimizer, or model capacity.

## 2. Seen-domain vs OOD validation

A seen-domain validation split was created by holding out trajectories from the training environments. The model was then evaluated intra-epoch on both:

- seen-val: held-out trajectories from train environments;
- OOD-val: held-out environment `seasidetown`.

During the first epoch, both seen-val and OOD-val improve substantially.

| step | epoch frac | seen global error | OOD global error | OOD gain | OOD cosine |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0000 | 0.031753 | 0.038254 | 0.000001 | 0.001759 |
| 100 | 0.1076 | 0.028016 | 0.035342 | 0.002912 | 0.239201 |
| 400 | 0.4306 | 0.025045 | 0.033252 | 0.005003 | 0.309982 |
| 900 | 0.9688 | 0.024103 | 0.032728 | 0.005527 | 0.330267 |
| 929 | 1.0000 | 0.023963 | 0.032741 | 0.005514 | 0.329265 |

The model is therefore not broken: the first epoch learns general predictive structure that transfers to the OOD validation environment.

## 3. Regularization sweep

A small regularization sweep was run with intra-epoch model selection.

| config | best OOD error | OOD gain | OOD cosine | best epoch frac |
| --- | ---: | ---: | ---: | ---: |
| base | 0.032728 | 0.005527 | 0.330267 | 0.9688 |
| dropout 0.15 + weight decay 0.01 | 0.032871 | 0.005384 | 0.324822 | 0.9688 |
| dropout 0.25 + weight decay 0.01 | 0.032872 | 0.005382 | 0.321655 | 0.9688 |

Simple dropout and weight decay do not improve OOD validation. They slightly hurt predictive performance in this setting.

## Interpretation

The repeated selection of epoch 1 is not caused by a basic training bug. The model learns normally and can overfit small batches. The issue is that after the first epoch, training continues to improve the training domains while OOD validation degrades, suggesting domain-specific overfitting.

Lowering the learning rate alone is not regularization. However, simple regularization through dropout and weight decay also does not solve the problem. This suggests that the remaining limitation is not merely optimizer instability, but the combination of:

1. strong domain shift between training environments and held-out environments;
2. limited number of independent environments;
3. validation based on a single OOD environment;
4. objective mismatch between global latent MSE and action-grounded prediction.

## Consequence

Future training should use:

- intra-epoch validation by default;
- separate seen-domain and OOD validation;
- model selection based on OOD global error and action-grounding diagnostics;
- final testing on `neighborhood` only after model selection;
- stronger protocol-level changes rather than only dropout/weight decay.

The current V-JEPA 2.1 result remains meaningful: V-JEPA 2.1 improves latent predictability and makes action conditioning measurably useful, but robust OOD action-grounding still requires a better protocol and/or more diverse action-conditioned data.

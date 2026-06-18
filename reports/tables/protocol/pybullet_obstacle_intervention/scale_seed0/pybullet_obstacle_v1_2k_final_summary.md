# PyBullet obstacle v1 2k scale-up final summary

## Goal

This experiment scales the PyBullet obstacle exact-intervention benchmark from 500 to 2000 groups in order to verify that the previous result was not a small-test artifact.

The benchmark uses true exact interventions:

```text
save simulator state with obstacles
execute action a_i
observe future_i
restore same simulator state
repeat for alternative actions
Dataset
quantity	value
groups	2000
candidates per group	5
samples	10000
chance top-1	0.200000
train / val / test groups	1600 / 200 / 200
latent shape	16 x 19
action dimension	2
image size	96
horizon	36
velocity	1.2
blocked directions per state	2
Oracle validation
model	alpha	top-1	mean rank	positive margin frac
identity	0.00	0.200000	3.000000	0.200000
delta oracle original	1.00	1.000000	1.000000	1.000000
delta oracle deranged	1.00	0.000000	3.494200	0.000000
random shuffle	1.00	0.197900	2.995100	0.197900

The oracle confirms that the scaled obstacle benchmark remains identifiable.

State-action ablation results
mode	best val MSE	test original top-1	test zero top-1	test shuffle top-1	test reverse top-1	original-zero gap	original-shuffle gap
full	0.00036601	0.960000	0.200000	0.189000	0.206000	0.760000	0.771000
action_only	0.00458287	0.210000	0.200000	0.197000	0.199000	0.010000	0.013000
state_only	0.00373666	0.200000	0.200000	0.200000	0.200000	0.000000	0.000000
no_context	0.00509070	0.200000	0.200000	0.200000	0.200000	0.000000	0.000000
Interpretation

The full state-action model reaches 0.960 top-1 on the test set, while every ablated model remains at chance.

This confirms that the benchmark cannot be solved by:

action only
state only
constant/no-context prediction

The task requires the joint pair:

current state + intervention action

The zero/shuffle


The task requires the joint pair:

```text
current state + intervention intervention tests also behave correctly: replacing the action by zero or by mismatched within-group actions collapses the trained full model back to chance.

## Comparison with previous stages

| protocol | oracle | learned full model | ablations |
| --- | --- | --- | --- |
| TartanAir latent-only silver | weak / invalid same-state | chance | not useful |
| TartanAir pose-aware silver | weak top-1 oracle | chance | not useful |
| Toy exact intervention | perfect | perfect | pass |
| PyBullet exact v0 | perfect | near-perfect | pass |
| PyBullet obstacle v1 500 | perfect | strong | pass |
| PyBullet obstacle v1 2k | perfect | strong, 0.960 test top-1 | pass |

## Scientific conclusion

The scale-up supports the central thesis:

```text
Action-grounded future prediction requires identifiable intervention structure.
Observational pseudo-counterfactual mining is insufficient.
Exact-intervention data makes action-grounded future ranking identifiable and learnable.

The PyBullet obstacle benchmark is now a validated minimal physical environment for testing action-grounded predictive state models.

Next step

The next limitation is that the current benchmark uses engineered patch tokens rather than a pretrained visual encoder.

Recommended next experiment:

PyBullet obstacle RGB encoder benchmark:
  save current/future RGB observations
  extract DINOv2 or V-JEPA features
  repeat oracle + full/action-only/state-only/no-context ablations

If the same pattern holds with pretrained visual representations, the result becomes much closer to the original SPSM motivation and much more publishable.

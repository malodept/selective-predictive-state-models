# PyBullet v0 state-action ablation final summary

## Goal

This ablation tests whether the PyBullet exact-intervention v0 benchmark is solved by true state-action grounding or merely by an action-template rule.

Four models are compared:

| mode | input |
| --- | --- |
| full | current state + action |
| action_only | action only |
| state_only | current state only |
| no_context | constant input |

## Results

| mode | best val MSE | test original top-1 | test zero top-1 | test shuffle top-1 | original-zero gap | original-shuffle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.00079370 | 0.992000 | 0.200000 | 0.160000 | 0.792000 | 0.832000 |
| action_only | 0.00749112 | 0.200000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 |
| state_only | 0.00624385 | 0.200000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 |
| no_context | 0.00850333 | 0.200000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 |

## Interpretation

The full model solves the benchmark almost perfectly, while all ablated models remain exactly at chance.

This shows that PyBullet v0 is not solved by action alone. The model needs the current state to localize the object and the action to select the correct intervention branch.

The intervention tests also behave correctly: when the trained full model is evaluated with zero or shuffled actions, top-1 collapses to chance or below chance.

## Scientific conclusion

PyBullet v0 validates a minimal but meaningful exact-intervention action-grounding benchmark:

```text
same simulator state
different actions
different futures
learned model succeeds only with correct state-action pair

This strengthens the core diagnosis of the project:

observational pseudo-counterfactual data is not enough;
exact-intervention data makes action-grounded future ranking identifiable and learnable.
Next step

PyBullet v0 is validated, but still physically simple. The next benchmark should introduce state-dependent dynamics:

PyBullet v1:
  random obstacles / walls
  collisions
  action effect depends on current state
  same exact-branch protocol

The expected test is again:

full >> action_only ≈ state_only ≈ chance

If full remains high under obstacles, then the benchmark becomes a stronger stepping stone toward CALVIN/Habitat.

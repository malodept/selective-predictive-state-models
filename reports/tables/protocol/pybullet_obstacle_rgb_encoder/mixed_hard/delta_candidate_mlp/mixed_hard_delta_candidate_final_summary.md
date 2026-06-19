# Mixed hard-negative delta-ranking final summary

## Goal

This experiment combines two complementary hard-negative structures in one exact-intervention ranking protocol:

```text
same-state / different-action negatives
same-action / different-state negatives

The objective is to test whether a model must use both intervention information and state-dependent visual dynamics. Ranking is performed in latent displacement space:

predicted_delta(anchor_state, action)
    versus
candidate_future - candidate_current

This avoids the absolute future shortcut observed in earlier candidate-ranking experiments.

Group mining
quantity	value
groups mined	1000
candidates per group	5
chance top-1	0.200000
same-state different-action negatives	2
same-action different-state negatives	2
max position distance	0.15
anchor blocked fraction	0.486000
off-diagonal blocked-mismatch fraction	0.802000
mean start-position distance	0.017410
mean DINOv2 delta distance	25.973200
Global top-1 results
mode	best val top-1	test original	action zero	state shuffle	state zero	gain vs chance
full	0.380000	0.390000	0.240000	0.260000	0.150000	0.190000
action_only	0.340000	0.310000	0.140000	0.310000	0.310000	0.110000
state_only	0.250000	0.250000	0.250000	0.250000	0.140000	0.050000
no_context	0.230000	0.270000	0.270000	0.270000	0.270000	0.070000
Pairwise decomposition by negative type
mode	same-state / different-action	same-action / different-state
full	0.650000	0.830000
action_only	0.970000	0.475000
state_only	0.560000	0.525000
no_context	0.515000	0.505000
Interpretation

The global top-1 ranking remains challenging, but the pairwise decomposition confirms the intended structure.

The action-only model performs very well against same-state / different-action negatives, but collapses to chance against same-action / different-state negatives. This shows that action-only can solve the action-confounding part of the protocol, but cannot solve the state-dependent dynamics part.

The full model is the only model that performs clearly above chance against both negative types:

same-state / different-action: 0.650
same-action / different-state: 0.830

This indicates that the full model uses both intervention information and state-dependent visual information.

Scientific conclusion

The mixed hard-negative protocol is not fully solved in global top-1 ranking, but it provides the cleanest evidence so far that frozen DINOv2 features support state-action dependent latent dynamics when the evaluation protocol removes the previous shortcuts.

Together, the experiments establish the following progression:

different-action exact groups:
    action information is useful, but action-only shortcuts exist

same-action delta hard negatives:
    state information is useful when the action shortcut is removed

mixed hard negatives:
    the full state-action model is the only model above chance against both action-confounding and state-confounding negatives

The next step is to improve the model architecture, not the basic protocol. A spatial transformer over DINOv2 tokens should be tested instead of a global MLP.

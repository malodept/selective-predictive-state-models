# Same-action hard-negative delta-ranking final summary

## Goal

The previous DINOv2 candidate-ranking protocol was solved by an action-template shortcut: candidates in each group corresponded to different actions, so an action-only model could recover the correct future without using the visual state.

This experiment removes that shortcut by constructing same-action hard-negative groups. All candidates in a group share the same action, and the ranking is performed in latent displacement space:

```text
predicted_delta(anchor_state, action)
    versus
candidate_future - candidate_current

This removes the absolute future shortcut and forces the model to reason about state-dependent dynamics, especially blocked versus unblocked motion.

Group mining
quantity	value
groups mined	1000
candidates per group	5
chance top-1	0.200000
max start-position distance	0.15
anchor blocked fraction	0.486000
off-diagonal blocked-mismatch fraction	0.999750
mean start-position distance	0.046039
mean DINOv2 future distance	38.675338
Results
mode	best val top-1	test original	action zero	state shuffle	state zero
full	0.900000	0.890000	0.890000	0.350000	0.330000
action_only	0.240000	0.270000	0.260000	0.270000	0.270000
state_only	0.910000	0.930000	0.930000	0.370000	0.330000
no_context	0.250000	0.260000	0.260000	0.260000	0.260000
Interpretation

The same-action delta-ranking protocol successfully breaks the previous action-template shortcut.

The action-only and no-context models remain close to chance, while the full and state-only models perform strongly. This shows that the DINOv2 representation contains state-dependent information that allows the model to distinguish blocked from unblocked latent dynamics.

State perturbation confirms this interpretation: shuffling or zeroing the state condition collapses full-model performance from 0.890 to approximately 0.33–0.35.

The fact that state_only is strong is expected in this protocol: all candidates within a group share the same action, so the test isolates the contribution of the visual state rather than the contribution of the action.

Scientific conclusion

This experiment provides the first strong DINOv2-based evidence that state-dependent visual information can support exact-intervention latent dynamics prediction.

Together with the earlier different-action experiments, the current result supports the project-level decomposition:

different-action exact groups:
    action information is necessary

same-action delta hard negatives:
    state information is necessary

combined:
    action-grounded predictive state modelling requires identifiable intervention structure
    and state-dependent visual dynamics
Next step

The next protocol should combine both axes in one benchmark:

mixed hard negatives:
    some candidates share the state but have different actions
    some candidates share the action but have different states / blocked conditions

In such a benchmark, action-only and state-only should both fail, while the full state-action model should succeed.

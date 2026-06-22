# Selective Predictive State Models under Controlled Interventions

# Selective Predictive State Models under Controlled Interventions

## Core idea

A world model should not only predict what will happen next. It should also know when its prediction is reliable, when the future state is identifiable from the available information, and when additional computation is worth paying for.

This project studies that question in a controlled latent-dynamics setting. We build an exact-intervention PyBullet benchmark where each simulator state is paired with multiple actions and future observations. Images are encoded with frozen DINOv2 patch tokens, and the learning problem is formulated as latent-displacement prediction: given the current visual state and an action, the model must identify the correct future latent displacement among hard negatives.

The central scientific question is:

> Can a predictive-state model learn action-grounded latent dynamics, remain reliable under controlled distribution shift, and selectively decide when expensive predictive computation is useful?

## v5.2 — Exact-intervention action grounding

The first key result is that a spatial action-conditioned Transformer learns the exact-intervention dynamics under a state-disjoint split. On the identifiable moving-action subset, the full model reaches near-perfect strict top-1 accuracy, around 0.998 across seeds.

This is important because the split prevents trivial memorization of simulator states. The hard-negative design also separates different failure modes: some negatives share the same state but differ in action, while others share the same action but differ in state. Ablations confirm that the full model is not solving the task through a single shortcut. Action-only, state-only, and no-context models each capture only partial structure, while the full state-action model combines both axes.

This establishes the first claim:

> Under exact interventions and state-disjoint evaluation, the model learns genuine state-action latent dynamics rather than exploiting a single shortcut.

## v5.3 — Controlled OOD shifts

The second step tests whether the trained model remains reliable under controlled distribution shifts. Without retraining, the v5.2 checkpoints are evaluated on new PyBullet variants that change horizon, velocity, and obstacle constraints.

The results show that the model is not fragile in a generic sense. It remains strong under horizon-only and velocity-only shifts. However, performance drops substantially when the number of blocked directions increases from two to three. This indicates that the main difficulty is not simply longer prediction or larger action magnitude, but more constrained environment geometry.

The failure decomposition is especially informative. Under block3 shifts, same-state/different-action discrimination remains relatively strong, while same-action/different-state discrimination degrades more. This suggests that action grounding survives the shift, but the model struggles with the more complex state geometry.

This establishes the second claim:

> Controlled OOD shifts reveal a specific failure mode: the model keeps action grounding but becomes less reliable when environment geometry becomes more constrained.

## v5.4–v5.5 — Why naive value-of-computation is not enough

The original SPSM motivation is selective computation: the system should decide when to use a cheap predictor and when to pay for a more expensive model.

Early routing experiments using incomplete-context cheap models — action-only, state-only, and no-context — show a large oracle gap. In theory, there is value in deciding when to call the full model. But in practice, these cheap models are too weak and incomplete. A confidence-threshold router tends to call the full model almost all the time, and a learned router mostly learns that the full model is usually better.

This is a useful negative result. It shows that action-only and state-only models are good scientific ablations, but poor cheap predictors for value-of-computation.

This motivates a better setup:

> The cheap model should be a smaller but complete state-action model, not an incomplete ablation.

## v5.6 — A meaningful cheap/expensive pair

A reduced full Transformer is trained with the same state-action inputs but lower capacity: model dimension 128, one layer, and four heads. This small-full model remains excellent on the ID benchmark and strong on several OOD variants.

Crucially, the expensive model is not uniformly better. On some hard OOD regimes, the small model is stronger; on others, especially block3 with longer horizon, the larger model helps. This creates a non-trivial value-of-computation problem.

The decision is no longer:

> Should we call the full model because it is always better?

It becomes:

> When does the expensive model provide a positive net gain over the cheap model, and when does it hurt or cost too much?

This is the first genuinely meaningful selective-computation setup in the project.

## v5.7–v5.8 — Context-aware value-of-computation

A learned gain router is trained to predict whether the expensive model improves over the small model. Using only cheap-model confidence and margin features gives a partial signal, but the router still fails under some hard OOD regimes. In particular, it routes too often to the expensive model on variants where the small model is actually safer.

Adding simple observable context features — horizon and velocity — improves the behavior. The context-aware router starts to route differently depending on the environment. On block3_h36, where the small model is stronger, it nearly avoids calling the expensive model. On block3_h72, where the expensive model helps more often, it routes more aggressively and improves over both cheap-only and confidence-threshold routing.

This is the strongest current result for the SPSM thesis:

> Value-of-computation is not captured by confidence alone. It depends on the interaction between model uncertainty and environment context.

## Current contribution

The project currently supports four contributions:

1. An exact-intervention latent-dynamics benchmark for testing action-grounded predictive-state models.
2. A state-disjoint hard-negative protocol that separates action grounding from state-geometry discrimination.
3. A controlled OOD analysis showing that constrained environment geometry creates a specific reliability failure mode.
4. A first value-of-computation study showing that context-aware routing can outperform naive confidence routing when deciding whether to use a larger predictive model.

## Current limitation

The context-aware router is still far from the oracle. It improves behavior qualitatively and sometimes quantitatively, but it does not fully identify when the larger model helps. The next step is to add explicit geometry or OOD-complexity features, and to evaluate routing under leave-one-shift-out generalization.

## Working title

Selective Predictive State Models: Reliability and Value-of-Computation under Controlled Interventions


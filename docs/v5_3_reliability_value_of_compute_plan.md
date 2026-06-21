# SPSM v5.3 — Reliability and value-of-computation extension

## Starting point

The v5.2 milestone established a state-disjoint exact-intervention PyBullet benchmark with mixed hard negatives and latent-displacement ranking.

The main empirical result is that a spatial action-conditioned delta Transformer reaches near-perfect moving-only strict top-1 under exact interventions, while action-only, state-only, and no-context ablations remain far below.

## Scientific goal

The v5.3 goal is to reconnect exact-intervention action grounding with the original SPSM thesis:

A latent predictive-state model should not only predict future latent states. It should also estimate when predictions are reliable, when the action-conditioned future is identifiable, and when additional computation is worth paying for.

## Core claims to test

1. Confidence separates identifiable from non-identifiable transitions.
2. Confidence remains calibrated under controlled distribution shift.
3. A learned reliability or gain predictor can improve deployment utility over cheap-only and always-expensive baselines.
4. Value-of-computation routing is stronger than generic uncertainty routing.

## Planned experiments

### E1 — Controlled OOD benchmark variants

Generate held-out PyBullet variants with changes in:

- obstacle layout,
- blocked direction distribution,
- object texture/color,
- camera or lighting perturbation,
- image noise or blur,
- horizon/velocity.

Keep the same exact-intervention structure and state-disjoint evaluation.

### E2 — Reliability diagnostics

Evaluate:

- confidence on moving vs stay actions,
- confidence vs strict top-1 correctness,
- confidence vs latent displacement error,
- ECE,
- Brier score,
- AUROC/AUPRC for failure prediction,
- coverage-risk curves.

### E3 — Cheap/expensive compute routing

Define:

- cheap model: smaller MLP or smaller Transformer,
- expensive model: current full spatial action-conditioned Transformer,
- oracle gain: utility(expensive) - utility(cheap) - compute cost,
- learned gain route: predict whether expensive compute is worth using.

### E4 — Routing baselines

Compare:

- cheap-only,
- expensive-only,
- random routing,
- confidence threshold,
- entropy/margin threshold,
- learned reliability head,
- learned gain head,
- oracle routing.

## Success criterion

The strongest v5.3 result would be:

Under exact-intervention and controlled OOD shifts, learned value-of-computation routing improves deployment utility over cheap-only and confidence-only routing, while maintaining calibrated reliability diagnostics.


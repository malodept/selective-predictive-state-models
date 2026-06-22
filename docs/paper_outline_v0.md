# Paper outline v0 — Selective Predictive State Models

## Working title

**Selective Predictive State Models: Reliability and Value-of-Computation under Controlled Interventions**

## One-sentence thesis

A predictive-state model should not only predict future latent states; it should also know when its prediction is reliable and when additional predictive computation is worth paying for.

## Abstract skeleton

World models are usually evaluated by prediction accuracy alone, but deployment requires knowing when predictions are reliable and when additional computation is useful. We introduce a controlled exact-intervention benchmark for latent predictive-state modeling. Each visual state is paired with multiple simulator interventions, encoded with frozen DINOv2 patch tokens, and evaluated through hard-negative latent-displacement ranking. A spatial action-conditioned Transformer learns state-action dynamics under state-disjoint splits, while ablations show that action-only and state-only models capture only partial structure. Controlled OOD shifts reveal that constrained environment geometry, not horizon or action magnitude alone, is the main source of reliability degradation. Finally, we study value-of-computation routing between a small full model and a larger full model. We show that confidence alone is insufficient, while simple context-aware routing better decides when expensive prediction helps or hurts.

## Core contributions

1. **Exact-intervention latent-dynamics benchmark**
   - PyBullet environment.
   - Multiple exact future interventions per state.
   - Frozen DINOv2 visual representation.
   - Latent-displacement ranking with hard negatives.

2. **State-disjoint action-grounding protocol**
   - Split by simulator states.
   - Mixed hard negatives.
   - Moving-only tie-aware evaluation.
   - Ablations: full, action-only, state-only, no-context.

3. **Controlled OOD reliability analysis**
   - Horizon shift.
   - Velocity shift.
   - Increased blocked-direction complexity.
   - Failure decomposition into action discrimination vs state-geometry discrimination.

4. **Value-of-computation routing**
   - Cheap model: small full Transformer.
   - Expensive model: larger full Transformer.
   - Oracle routing analysis.
   - Confidence threshold baseline.
   - Learned gain router.
   - Context-aware gain router.

## Paper structure

## 1. Introduction

### Goal

Motivate the gap between prediction and deployment:

- World models are often judged by prediction quality.
- Real systems need reliability estimation.
- Compute is not free.
- A useful model should decide when to trust cheap prediction and when to pay for expensive prediction.

### Main research question

Can a latent predictive-state model:

1. learn action-grounded state transitions under exact interventions,
2. detect reliability degradation under controlled OOD shifts,
3. estimate the value of additional predictive computation?

### Main empirical story

- v5.2: action-grounded prediction works under state-disjoint exact interventions.
- v5.3: OOD shifts reveal geometry-specific failure modes.
- v5.6–v5.8: selective computation becomes non-trivial when cheap and expensive models are both competent but not uniformly ordered.

## 2. Related work

### Areas to cover

1. World models and predictive-state representations.
2. Joint embedding predictive architectures / latent prediction.
3. Counterfactual and intervention-based evaluation.
4. Selective prediction, confidence calibration, risk-coverage.
5. Conditional computation / adaptive computation / value-of-computation.

### Positioning

This work is not trying to beat a benchmark with a large model. It studies when latent predictions are identifiable, reliable, and worth computing.

## 3. Benchmark and protocol

### 3.1 Exact-intervention environment

Describe:

- PyBullet scene.
- RGB observations.
- Discrete action set.
- Exact reset of simulator state.
- Future observation per action.
- Stay actions are non-identifiable in latent displacement and handled separately.

### 3.2 Representation

- Frozen DINOv2 ViT-S/14.
- Patch tokens.
- Pooling from 16×16 to 4×4.
- Future prediction is evaluated in latent-token displacement space.

### 3.3 Hard-negative ranking task

For each anchor:

- positive = correct future latent displacement;
- same-state/different-action negatives;
- same-action/different-state negatives;
- possibly mixed negatives.

Metrics:

- strict top-1;
- tie-aware top-1;
- pairwise margin decomposition;
- moving-only evaluation.

### 3.4 State-disjoint split

State-disjoint split prevents memorization of exact simulator states.

## 4. Action-grounded latent dynamics

### Claim

The full state-action Transformer learns genuine state-action latent dynamics.

### Main table

**Table 1 — ID performance and ablations**

Rows:

- full;
- action-only;
- state-only;
- no-context.

Columns:

- moving-only strict top-1;
- same-state/different-action;
- same-action/different-state.

Expected message:

- full combines both axes;
- action-only handles action axis;
- state-only handles state axis;
- no-context fails.

### Main result

Full model reaches near-perfect moving-only strict top-1 under state-disjoint exact-intervention evaluation.

## 5. Controlled OOD reliability

### Claim

The model is not generically fragile; it fails under specific geometry-complexity shifts.

### OOD variants

- ID: block2, horizon 36, velocity 1.2.
- block2_h72: horizon shift.
- block2_v18: velocity shift.
- block3_h36: increased blocked-direction complexity.
- block3_h48.
- block3_h72.

### Main table

**Table 2 — Controlled OOD performance**

Columns:

- variant;
- full moving-only top-1;
- same-action/diff-state;
- same-state/diff-action;
- confidence/risk-coverage maybe.

Main message:

- horizon alone: limited degradation;
- velocity alone: limited degradation;
- block3: strong degradation;
- failure is mostly same-action/different-state.

### Figure

**Figure 1 — OOD performance by shift type**

Bar plot:

- ID;
- horizon shift;
- velocity shift;
- block3 shifts.

## 6. Value-of-computation setup

### Problem

Given:

- cheap model prediction;
- expensive model prediction;
- compute cost lambda;

route to expensive model only when:

`correct_full - correct_cheap > lambda`.

### Cheap/expensive pair

Bad cheap models:

- action-only;
- state-only;
- no-context.

Useful negative result:

- they are scientific ablations, not good cheap predictors.

Better cheap model:

- small full Transformer;
- model_dim=128;
- layers=1;
- heads=4.

Expensive model:

- full Transformer;
- model_dim=384;
- layers=3;
- heads=6.

### Main table

**Table 3 — Small full vs full**

Columns:

- variant;
- cheap accuracy;
- full accuracy;
- oracle utility;
- oracle route rate.

Message:

- full is not uniformly better;
- small_full is sometimes better;
- VoC is non-trivial.

## 7. Routing experiments

### Baselines

- cheap-only;
- full-only;
- oracle routing;
- confidence threshold;
- learned gain router;
- context-aware learned gain router.

### Main result

Context-aware routing improves over confidence threshold in key regimes.

Important cases:

- block3_h36: small model is safer; router should avoid full.
- block3_h72: full helps; router should route more.

### Main table

**Table 4 — Value-of-computation routing**

Rows:

- cheap-only;
- full-only;
- confidence threshold;
- learned gain;
- context-aware learned gain;
- oracle.

Columns:

- utility at lambda = 0.10 or 0.30;
- route rate;
- accuracy;
- maybe harmful route rate.

### Main figure

**Figure 2 — Utility vs lambda**

Curves:

- cheap-only;
- full-only;
- confidence;
- context-aware;
- oracle.

Separate panels:

- block3_h36;
- block3_h72.

## 8. Error analysis

### Claim

The remaining gap to oracle comes from missing geometry/OOD-complexity features.

Use gain-router error audit:

- good available;
- route precision;
- route recall;
- harmful route;
- wasteful route;
- missed good.

### Message

Confidence and simple context help, but routing still misses many positive-gain examples and sometimes routes harmful examples. More explicit state-geometry features are needed.

## 9. Limitations

1. PyBullet benchmark is controlled and synthetic.
2. Frozen DINOv2 representation may not be optimal for physical dynamics.
3. Only discrete actions tested.
4. OOD variants are still limited.
5. Router uses simple metadata; no explicit geometry descriptor yet.
6. Expensive model is not always more robust, which is interesting but complicates the narrative.

## 10. Future work

1. Add geometry-complexity features.
2. Leave-one-shift-out generalization.
3. More seeds for small_full and router.
4. Harder environments.
5. Continuous actions.
6. Move from ranking to planning/control utility.
7. Test JEPA/V-JEPA-style representations.
8. Apply to remote sensing / Earth observation later, but only after the core claim is stable.

## Required figures and tables

### Figures

1. Benchmark schematic: exact intervention, multiple futures per state.
2. OOD performance by shift type.
3. Risk-coverage / confidence under OOD.
4. Value-of-computation utility vs lambda.
5. Routing behavior on block3_h36 vs block3_h72.

### Tables

1. ID action-grounding ablations.
2. OOD variants summary.
3. Small_full vs full.
4. Routing comparison.
5. Error audit.

## Immediate next tasks

1. Generate paper-ready CSV summaries from existing markdown tables.
2. Generate figures from those summaries.
3. Create a short LaTeX paper skeleton.
4. Write Section 3 first, because the benchmark/protocol is the scientific foundation.
5. Only then decide whether more experiments are needed.


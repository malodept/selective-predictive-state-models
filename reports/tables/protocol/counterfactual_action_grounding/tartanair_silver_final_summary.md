# TartanAir silver counterfactual protocol final summary

This report summarizes the TartanAir silver matched-state experiments.

## Motivation

The goal was to test whether TartanAir can provide pseudo-counterfactual action-grounded groups of the form:

```text
near-same current state
different actions
different reachable futures

This would allow evaluating whether an action-conditioned latent predictor can rank the future caused by the correct action above futures caused by alternative actions.

Latent-only silver groups

The first miner selected neighbours using V-JEPA latent proximity and action/future separation.

It produced enough candidate groups, but the pose audit showed that latent proximity did not imply physical state proximity.

split	median position distance	median velocity distance	median quaternion distance
train	9.404141	0.556485	0.205726
val	16.940291	1.007366	0.203514
test	75.526161	0.940008	0.214391

The test candidates were often tens of meters away from the anchor. Therefore, latent-only groups were not valid near-same-state groups.

Strictly filtering the closest latent groups did not reveal hidden action-grounding signal. V-JEPA MSE stayed at chance on the closest 5%, 10%, and 20% of test groups.

Pose-aware silver groups

A pose-aware miner was then introduced using:

same environment;
different trajectory;
position distance constraint;
velocity distance constraint;
quaternion distance constraint;
action distance lower bound;
future distance lower bound.

This substantially improved physical matching.

protocol	median position distance	median velocity distance	median quaternion distance
latent-only test	75.526161	0.940008	0.214391
pose-aware v0 test	2.717982	0.565429	0.070579
pose-aware strict test	1.395376	0.332018	0.019905
Model evaluation

Despite improved physical matching, existing predictors stayed at chance on pose-aware groups.

model	protocol	alpha	original top-1	zero top-1	shuffle top-1
V-JEPA MSE	pose-aware v0 pilot	1.00	0.200000	0.200000	0.200000
V-JEPA candidate λ=0.2	pose-aware v0 pilot	1.00	0.200000	0.200000	0.200000
Delta-transfer oracle

To check whether the benchmark itself was identifiable, a delta-transfer oracle was evaluated:

prediction_k = anchor_current + alpha * (candidate_future_k - candidate_current_k)

This oracle uses the true latent displacement of each candidate and applies it to the anchor.

protocol	groups	alpha	oracle top-1	mean rank	positive margin frac
pose-aware v0 pilot	500	1.00	0.205600	2.545200	0.204800
pose-aware strict	168	1.00	0.214286	2.282143	0.213095

The strict oracle improves mean rank, but top-1 remains only slightly above chance.

Interpretation

TartanAir observational silver groups are not sufficiently identifiable for robust action-grounded future ranking.

The key finding is not simply that the learned models fail. Even an optimistic delta-transfer oracle barely exceeds chance top-1. This indicates that the pseudo-counterfactual groups do not reliably isolate the effect of action.

Decision

Do not train on TartanAir silver groups.

The project should move to exact-intervention environments where multiple futures can be generated from the same simulator state under different actions.

Recommended next step:

CALVIN or Habitat exact-branch protocol

The next benchmark should be based on true interventions:

save simulator state
execute action a_1 -> future_1
reset same simulator state
execute action a_2 -> future_2
...
evaluate whether model ranks future_i highest under action a_i
Scientific conclusion

The bottleneck is counterfactual identifiability. Frozen V-JEPA improves temporal prediction, and action conditioning is weakly measurable, but observational TartanAir mining does not provide the causal supervision needed for robust action-grounded evaluation.

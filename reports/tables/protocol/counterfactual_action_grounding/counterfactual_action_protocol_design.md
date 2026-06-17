# Counterfactual action-grounding protocol design

## Core diagnosis

The current TartanAir environment-split protocol is not a definitive action-grounding benchmark. It is a useful stress test, but it does not provide exact counterfactual supervision.

The main empirical finding so far is:

- DINOv2 patch tokens improve latent prediction compared with pooled features, but action-grounding remains weak.
- V-JEPA 2.1 frozen video tokens make latent future prediction much stronger.
- V-JEPA 2.1 makes action conditioning measurably useful beyond latent video flow.
- The candidate objective increases action sensitivity, but does not produce robust action-grounded candidate matching.
- Therefore, action sensitivity is not sufficient: the model can move more when action changes without moving toward the correct counterfactual future.

The bottleneck is now primarily protocol/data identification, not simply encoder choice or loss design.

## Current failure mode

The current protocol mixes several effects:

1. Representation quality.
2. Predictor capacity.
3. Action usage.
4. Domain shift.
5. Candidate construction quality.
6. Counterfactual identifiability.

Because current candidates are not exact branches from the same physical state under different actions, a candidate loss can reward sensitivity without grounding. This explains why the candidate objective increased prediction spread while top-1 candidate matching stayed near chance.

## Principle

A rigorous action-grounding benchmark must contain candidate groups of the form:

```text
same current state
different actions
different reachable futures

The key benchmark should not merely ask:

does the prediction change when action changes?

It should ask:

does the prediction produced under action a_i rank the future actually caused by a_i above futures caused by other actions?
Protocol ladder
Bronze: current diagnostics

Keep the current checks:

original action
zero action
shuffled action
reversed action
alpha sweep
action sensitivity spread ratio
candidate top-1 / margin diagnostics

These are diagnostics only. They do not prove counterfactual grounding.

Silver: matched-state pseudo-counterfactuals in TartanAir

Build pseudo-counterfactual candidate groups by mining near-identical current states with different actions.

For each anchor sample:

(z_t, pose_t, velocity_t, action_t, z_future_t)

retrieve neighbours with:

similar current latent state
similar pose / local geometry
different action
plausibly different future

Use short horizons first:

h ∈ {1, 2, 4, 8}
K = 5 candidates

Quality control:

max latent distance
max pose distance
max velocity distance
min action distance
min future separation
no same-trajectory leakage unless deliberately allowed
environment-stratified reporting

This improves the current TartanAir benchmark but does not fully solve confounding.

Gold: exact simulator branches

The clean scientific test requires exact interventions.

For a simulator state s_t:

save simulator state
execute action a_1 for h steps -> future_1
reset to same simulator state
execute action a_2 for h steps -> future_2
...
execute action a_K for h steps -> future_K

Then evaluate whether the model ranks future_i highest under action a_i.

Recommended exact-intervention simulators:

CALVIN first.
Habitat second.
CausalWorld if a minimal causal-control benchmark is useful.
Dataset priority
priority	dataset/protocol	role
P0	Current TartanAir env split	freeze as negative-but-informative baseline
P1	TartanAir silver matched-state mining	fast next experiment; improves candidate construction
P2	CALVIN exact branches	cleanest scientific action-grounding test
P3	Habitat exact branches	richer embodied/navigation/rearrangement test
P4	TartanAir gold subset	exact branches if simulator reset access is feasible
P5	BridgeData / DROID / TartanDrive	realism transfer after exact simulator success
Metrics

Report three distinct families of metrics.

Prediction quality
identity error
raw error
calibrated global error
gain over identity
cosine between predicted and true latent displacement
Action sensitivity
prediction spread across candidate actions
true future spread
spread ratio
original vs zero vs shuffled action outputs
Action-grounded identification
candidate top-1 / top-k accuracy
mean correct rank
diagonal MSE vs best off-diagonal MSE
margin and positive-margin fraction
intervention gap:
top1(original) - top1(shuffle)
top1(original) - top1(zero)
OOD gap between seen-val and OOD-val/test
Alpha sweep as first-class evaluation

Do not only report one calibrated alpha.

Report:

alpha_MSE*    = alpha chosen to minimize global prediction error
alpha_rank*   = alpha chosen to maximize top-1 / rank metric
alpha = 1     = raw model displacement
AUC_top1      = area under top1(alpha)
calibration gap = top1(alpha_rank*) - top1(alpha_MSE*)

This is necessary because MSE calibration can suppress action-grounded ranking signal.

Statistical discipline

For every candidate-matching result:

report trajectory-stratified or environment-stratified bootstrap confidence intervals;
use paired bootstrap/permutation tests for original vs zero/shuffle;
report seed variance separately from data uncertainty;
do not claim success from a single held-out environment.

Minimum pilot standard:

8000 candidate groups per split
3 seeds

Main claim standard:

20000 candidate groups per split
5 seeds
at least 3 OOD validation environments
at least 3 OOD test environments
Success thresholds

A candidate/ranking objective should only be continued if:

top1(original, alpha=1) >= chance + 3 percentage points
and
top1(original) - top1(shuffle) >= 2 percentage points
and
confidence interval excludes zero

A result that improves spread ratio but not intervention gap is classified as:

non-grounded action sensitivity

not as action grounding.

Next implementation step

The next concrete branch task is P1:

implement TartanAir silver matched-state group mining

This should be done before any new heavy training run.

Initial target:

script: scripts/mine_matched_state_groups.py
input:  outputs/tartanair_vjepa2_1_base384_causal16_multi_env/env_split/features_*.npz
output: outputs/counterfactual/tartanair_silver/*_groups.npz

The first version should only build and audit groups. No training yet.

Required group fields:

anchor_index
candidate_indices
correct_candidate
anchor_env
anchor_trajectory
candidate_envs
candidate_trajectories
latent_distances
pose_distances
action_distances
future_distances
horizon

Required QC report:

number of candidate groups
latent distance distribution
action distance distribution
future distance distribution
same-trajectory fraction
same-environment fraction
candidate diversity
chance baseline
Decision logic

If TartanAir silver improves original-vs-shuffle on easy subsets, continue to silver training.

If silver still fails, prioritize CALVIN exact branches rather than more loss engineering.

If CALVIN exact branches succeed but TartanAir silver fails, the bottleneck is TartanAir candidate construction.

If CALVIN exact branches fail, revisit architecture, action representation, horizon, and whether the frozen encoder is sufficient.

Bottom line

The project should now move from:

better encoder / better loss

to:

better identification of action effects

The next scientific milestone is not a lower MSE. It is a benchmark where action-grounded future ranking is identifiable.

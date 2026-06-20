# Scale-5k mixed hard-negative delta Transformer final summary

## Goal

This experiment scales the DINOv2 mixed hard-negative protocol from the 500-group pilot to a 5000-group RGB PyBullet exact-intervention dataset.

The benchmark uses mixed hard negatives:

```text
same-state / different-action negatives
same-action / different-state negatives

Ranking is performed in latent displacement space:

predicted_delta(anchor_state, action)
    versus
candidate_future - candidate_current

This removes the absolute-future shortcut and tests whether the model uses both action information and state-dependent visual dynamics.

Dataset and representation
quantity	value
RGB exact-intervention groups	5000
samples	25000
DINOv2 raw tokens	25000 x 256 x 384
DINOv2 pooled tokens	25000 x 16 x 384
mixed hard groups mined	10000
train / val / test groups	8000 / 1000 / 1000
candidates per group	5
chance top-1	0.200000
Mixed hard group mining
quantity	value
same-state different-action negatives	2
same-action different-state negatives	2
max position distance	0.15
anchor blocked fraction	0.500700
offdiag blocked-mismatch fraction	0.812900
mean start-position distance	0.005271
mean DINOv2 delta distance	25.572472
Global top-1 results
mode	best val top-1	test original	action zero	state shuffle	state zero
full	1.000000	0.998000	0.360000	0.472000	0.405000
action_only	0.369000	0.362000	0.176000	0.362000	0.362000
state_only	0.373000	0.358000	0.358000	0.172000	0.151000
no_context	0.210000	0.222000	0.222000	0.222000	0.222000
Pairwise decomposition by negative type
mode	same-state / different-action	same-action / different-state
full	0.998500	1.000000
action_only	0.939000	0.496500
state_only	0.568000	0.940500
no_context	0.530000	0.511000
Interpretation

The scaled benchmark confirms the pilot result on a much larger test set.

The full spatial Transformer solves the mixed hard-negative protocol almost perfectly, with test top-1 = 0.998 on 1000 held-out groups. The ablations remain far below the full model.

The pairwise audit provides the cleanest interpretation. The action-only model is strong against same-state / different-action negatives, but remains at chance against same-action / different-state negatives. The state-only model shows the complementary pattern. The no-context model remains near chance on both axes. The full model is the only model that solves both axes.

Scientific conclusion

The scale-5k result is now the main result of the project.

Frozen DINOv2 patch-token features contain sufficient information for state-action dependent exact-intervention latent dynamics. A spatial action-conditioned Transformer can exploit this information at scale, while action-only, state-only, and no-context baselines cannot solve the mixed hard-negative protocol.

This supports the central claim:

Under an identifiable exact-intervention protocol,
a spatial action-conditioned Transformer learns state-action dependent latent dynamics
from frozen visual representations.


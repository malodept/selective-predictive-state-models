# Mixed hard-negative delta Transformer final summary

## Goal

This experiment tests whether a model can learn state-action dependent latent dynamics from frozen DINOv2 visual features under an exact-intervention protocol.

The benchmark uses mixed hard negatives:

```text
same-state / different-action negatives
same-action / different-state negatives

Ranking is performed in latent displacement space:

predicted_delta(anchor_state, action)
    versus
candidate_future - candidate_current

This removes the absolute-future shortcut and requires the model to predict the correct latent displacement.

Representation and model
quantity	value
visual encoder	frozen DINOv2 ViT-S/14
token pooling	16x16 to 4x4
tokens	16
token dimension	384
model	spatial Transformer
Transformer layers	3
attention heads	6
model dimension	384
candidates per group	5
chance top-1	0.200000
Global top-1 results
mode	best val top-1	test original	action zero	state shuffle	state zero	gain vs chance
full	0.980000	0.970000	0.200000	0.330000	0.320000	0.770000
action_only	0.310000	0.280000	0.280000	0.280000	0.280000	0.080000
state_only	0.310000	0.250000	0.250000	0.160000	0.110000	0.050000
no_context	0.220000	0.210000	0.210000	0.210000	0.210000	0.010000
Pairwise decomposition by negative type
mode	same-state / different-action	same-action / different-state
full	0.985000	0.980000
action_only	0.755000	0.515000
state_only	0.555000	0.760000
no_context	0.525000	0.505000
Interpretation

The full Transformer solves the mixed hard-negative protocol almost perfectly, with test top-1 = 0.970.

The perturbation tests confirm that both inputs are necessary:

full original      = 0.970
full action zero   = 0.200
full state shuffle = 0.330
full state zero    = 0.320

Zeroing the action collapses the full model to chance. Shuffling or zeroing the state also causes a large collapse. Therefore, the model is not relying on a pure action template or a pure state shortcut.

The pairwise audit confirms the intended decomposition. The action-only model performs above chance against same-state / different-action negatives, but remains near chance against same-action / different-state negatives. The state-only model shows the complementary behavior. The no-context model remains near chance on both. The full model is the only one that performs near-perfectly against both negative types.

Scientific conclusion

This is the cleanest result of the project so far.

Frozen DINOv2 patch-token features contain sufficient information for exact-intervention latent dynamics. However, a global MLP is not enough to exploit this structure on mixed hard negatives. A spatial Transformer over DINOv2 tokens succeeds, showing that the architecture must preserve and process spatial token structure.

The result supports the following claim:

Under an identifiable exact-intervention protocol, a spatial action-conditioned Transformer can learn state-action dependent latent dynamics from frozen visual representations.
Progression of evidence
stage	conclusion
TartanAir pseudo-counterfactual mining	not identifiable enough
PyBullet exact intervention with engineered tokens	protocol sanity check passes
DINOv2 different-action ranking	action information is exploitable, but action-only shortcuts exist
DINOv2 same-action delta ranking	state information is exploitable when action shortcuts are removed
DINOv2 mixed hard negatives with MLP	protocol is meaningful, but MLP is weak
DINOv2 mixed hard negatives with spatial Transformer	full state-action model succeeds cleanly
Next steps

The next scientific step is not another protocol patch. The protocol is now strong enough. The next steps are:

repeat the Transformer result over multiple seeds;
scale from 500 RGB groups / 1000 mined groups to a larger dataset;
compare DINOv2 against V-JEPA features;
test whether the same conclusion holds on a more realistic simulator or robot dataset.

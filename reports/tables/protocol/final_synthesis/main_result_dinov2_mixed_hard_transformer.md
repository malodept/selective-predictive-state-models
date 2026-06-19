# Main result: DINOv2 mixed hard-negative spatial Transformer

## Core claim

Under an identifiable exact-intervention protocol, frozen DINOv2 patch-token features support state-action dependent latent dynamics prediction when the model architecture preserves spatial token structure.

## Final benchmark

The final protocol uses mixed hard negatives:

```text
same-state / different-action negatives
same-action / different-state negatives

Ranking is performed in latent displacement space:

predicted_delta(anchor_state, action)
    versus
candidate_future - candidate_current

This removes the absolute-future shortcut and forces the model to combine action information with state-dependent visual dynamics.

Representation and model
component	value
visual encoder	frozen DINOv2 ViT-S/14
token grid	pooled from 16x16 to 4x4
tokens	16
token dimension	384
dynamics model	spatial Transformer
Transformer depth	3 layers
attention heads	6
model dimension	384
candidates per group	5
chance top-1	0.200
Main multiseed result

The full state-action Transformer is robust across seeds 0, 1, and 2. It remains far above all ablations.

Expected headline result:

full        ≈ 0.95 top-1
action_only ≈ 0.29 top-1
state_only  ≈ 0.25 top-1
no_context  ≈ 0.19 top-1
Interpretation

The full model strongly outperforms all ablations. Zeroing the action or corrupting the visual state sharply reduces performance. This shows that the model is not solving the benchmark through a pure action template, a pure state shortcut, or a no-context prior.

The pairwise decomposition further validates the protocol:

same-state / different-action negatives:
    tests action information

same-action / different-state negatives:
    tests state-dependent visual dynamics

The full model is the only model that performs clearly above chance on both axes.

Scientific conclusion

The project now has a controlled positive result:

Frozen DINOv2 features contain sufficient information for exact-intervention latent dynamics.
A global MLP fails to exploit this structure on mixed hard negatives.
A spatial action-conditioned Transformer succeeds.

This supports the hypothesis that action-grounded predictive state modelling requires both an identifiable intervention protocol and an architecture that preserves spatial visual structure.

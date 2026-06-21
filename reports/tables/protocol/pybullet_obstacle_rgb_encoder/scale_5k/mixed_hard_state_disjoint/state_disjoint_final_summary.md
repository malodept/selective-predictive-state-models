# State-disjoint mixed hard benchmark final summary

## Main result

This is the strongest current evaluation of the project.

The benchmark uses:

- exact-intervention PyBullet rollouts;
- frozen DINOv2 visual representations;
- 5000 simulator states;
- state-disjoint train/validation/test splits;
- mixed hard negatives;
- latent-displacement ranking;
- tie-aware evaluation;
- a moving-only identifiable test subset, excluding the non-identifiable `stay` action.

## Dataset and split

| quantity | value |
| --- | ---: |
| RGB exact-intervention states | 5000 |
| samples | 25000 |
| DINOv2 pooled tokens | 25000 x 16 x 384 |
| train / val / test simulator states | 4000 / 500 / 500 |
| train / val / test mined groups | 8000 / 1000 / 1000 |
| candidates per group | 5 |
| chance top-1 | 0.200000 |
| state overlap train/val/test | 0 / 0 / 0 |

## Why `stay` is excluded for the strongest metric

In latent-displacement space, the `stay` action is non-identifiable across states:

```text
delta(state A, stay) = 0
delta(state B, stay) = 0

The identifiability audit confirmed that all same-action/different-state stay pairs have exact zero delta distance. These cases produce exact ties that no delta-prediction model can resolve. Therefore the strongest metric is computed on the identifiable moving-action subset.

Moving-only identifiable evaluation
model	strict top-1	tie-aware top-1
full Transformer, seed 0	0.998779	0.998779
full Transformer, seed 1	0.997558	0.997558
full Transformer, seed 2	0.996337	0.996337
full Transformer, mean ± std	0.997558 ± 0.001221	0.997558 ± 0.001221
Full Transformer pairwise multiseed summary
metric	mean ± std
same-state / different-action	0.999389 ± 0.000000
same-action / different-state	0.998982 ± 0.001271
Single-seed ablation comparison on moving-only test subset
mode	strict top-1	same-state / different-action	same-action / different-state
full	0.998779	0.999389	1.000000
action_only	0.327228	0.902930	0.495726
state_only	0.343101	0.498168	0.911477
no_context	0.191697	0.442002	0.529915
Interpretation

The full spatial action-conditioned Transformer solves the identifiable state-disjoint benchmark almost perfectly.

The ablations give the key mechanistic interpretation:

action_only can distinguish different actions from the same state, but fails when the action is fixed and the state changes;
state_only can distinguish state-dependent dynamics, but fails when the state is fixed and the action changes;
no_context remains near chance;
only the full model solves both axes.
Scientific conclusion

The final result supports the central claim:

A spatial action-conditioned Transformer can learn state-action dependent latent dynamics
from frozen DINOv2 visual representations under an exact-intervention,
state-disjoint, identifiable benchmark.

This result is not explained by simulator-state leakage, action-only shortcuts, state-only shortcuts, no-context dataset bias, or argmin tie-breaking artifacts.

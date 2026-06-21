# State-disjoint mixed hard benchmark final summary

## Goal

This experiment evaluates whether a spatial action-conditioned Transformer can learn state-action dependent latent dynamics under a strict state-disjoint protocol.

Unlike the previous random mined-group split, this benchmark first separates simulator states into train, validation, and test splits before hard-negative mining. Therefore, no simulator state appears in more than one split.

## Dataset

| quantity | value |
| --- | ---: |
| RGB exact-intervention states | 5000 |
| samples | 25000 |
| DINOv2 pooled tokens | 25000 x 16 x 384 |
| train / val / test states | 4000 / 500 / 500 |
| train / val / test mined groups | 8000 / 1000 / 1000 |
| candidates per group | 5 |
| chance top-1 | 0.200000 |
| state overlap train/val/test | 0 / 0 / 0 |

## Why tie-aware evaluation is needed

The all-action state-disjoint benchmark includes the `stay` action. In latent-displacement space, `stay` is non-identifiable across states because:

```text
delta(state A, stay) = 0
delta(state B, stay) = 0

The action-identifiability audit confirmed this exactly: all same-action/different-state stay pairs have exact zero delta distance. Therefore, biased argmin top-1 can overestimate performance because the correct candidate is stored in column 0.

All-action tie-aware metrics
mode	biased argmin top-1	strict top-1	tie-aware top-1
full	0.999000	0.818000	0.878333
action_only	0.449000	0.268000	0.328333
state_only	0.339000	0.281000	0.300333
no_context	0.263000	0.157000	0.192333

The full model never loses against same-action/different-state negatives in the all-action audit; the apparent strict drop is due to exact ties caused by stay.

Moving-only identifiable evaluation

After removing non-identifiable stay anchors, the benchmark becomes fully identifiable in latent-displacement space.

mode	strict top-1	tie-aware top-1	mean tie count
full	0.998779	0.998779	1.000000
action_only	0.327228	0.327228	1.000000
state_only	0.343101	0.343101	1.000000
no_context	0.191697	0.191697	1.000000
Moving-only pairwise decomposition
mode	same-state / different-action	same-action / different-state
full	0.999389	1.000000
action_only	0.902930	0.495726
state_only	0.498168	0.911477
no_context	0.442002	0.529915
Scientific conclusion

The state-disjoint benchmark confirms that the result is not explained by train/test simulator-state leakage.

On the identifiable moving-action subset, the full spatial action-conditioned Transformer reaches strict top-1 = 0.998779, while action-only, state-only, and no-context baselines remain far below it.

The pairwise decomposition gives the cleanest interpretation:

action-only solves same-state/different-action negatives but fails on same-action/different-state negatives;
state-only solves same-action/different-state negatives but fails on same-state/different-action negatives;
no-context remains near chance;
the full model solves both axes almost perfectly.

This supports the central claim:

A spatial action-conditioned Transformer can learn state-action dependent latent dynamics from frozen DINOv2 visual representations under an exact-intervention, state-disjoint, identifiable benchmark.


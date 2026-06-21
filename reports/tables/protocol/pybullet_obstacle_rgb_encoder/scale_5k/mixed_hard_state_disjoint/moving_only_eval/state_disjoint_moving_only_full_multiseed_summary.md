# State-disjoint moving-only full Transformer multiseed summary

Values are mean ± sample standard deviation over seeds 0, 1, 2.

| model | strict top-1 | tie-aware top-1 | same-state / different-action | same-action / different-state |
| --- | ---: | ---: | ---: | ---: |
| full Transformer | 0.997558 ± 0.001221 | 0.997558 ± 0.001221 | 0.999389 ± 0.000000 | 0.998982 ± 0.001271 |

## Interpretation

This is the strongest current evaluation: state-disjoint splits, non-identifiable `stay` anchors removed, and tie-aware reporting.
The full Transformer should remain near-perfect across seeds if the state-action grounding result is robust.

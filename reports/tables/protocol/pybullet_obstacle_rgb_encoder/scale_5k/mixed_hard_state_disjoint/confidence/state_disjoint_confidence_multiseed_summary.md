# State-disjoint confidence multiseed summary

Values are mean ± sample standard deviation over seeds 0, 1, 2.

## Global reliability metrics

| metric | value |
| --- | ---: |
| mean confidence | 0.873046 ± 0.000465 |
| biased top-1 | 0.998000 ± 0.001000 |
| strict top-1 | 0.817000 ± 0.001000 |
| tie-aware top-1 | 0.877333 ± 0.001000 |
| correct tied with another candidate | 0.181000 ± 0.000000 |
| ECE vs tie-aware target | 0.005377 ± 0.001102 |
| ECE vs strict target | 0.064962 ± 0.001156 |

## Selective prediction curve

| coverage | confidence | strict top-1 | tie-aware top-1 | stay fraction |
| ---: | ---: | ---: | ---: | ---: |
| 1.00 | 0.873046 ± 0.000465 | 0.817000 ± 0.001000 | 0.877333 ± 0.001000 | 0.181000 ± 0.000000 |
| 0.90 | 0.933402 ± 0.000492 | 0.907778 ± 0.001111 | 0.937778 ± 0.001111 | 0.090000 ± 0.000000 |
| 0.80 | 0.996932 ± 0.000271 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| 0.70 | 0.999172 ± 0.000079 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| 0.60 | 0.999621 ± 0.000038 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| 0.50 | 0.999791 ± 0.000025 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| 0.40 | 0.999876 ± 0.000016 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 |
| 0.30 | 0.999926 ± 0.000010 | 1.000000 ± 0.000000 | 1.000000 ± 0.000000 | 0.000000 ± 0.000000 |

## Action confidence separation

| action subset | mean confidence |
| --- | ---: |
| stay | 0.331265 ± 0.000172 |
| moving actions | 0.992722 ± 0.002975 |

## Interpretation

The full Transformer assigns low confidence to the non-identifiable `stay` cases and high confidence to identifiable moving actions.
At 80% coverage, the retained set contains no `stay` anchors and reaches perfect strict and tie-aware accuracy across all three seeds.
This reconnects the exact-intervention benchmark to the original SPSM reliability goal: confidence is informative about when a latent prediction is identifiable and trustworthy.

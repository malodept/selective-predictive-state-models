# SPSM v24 measured capacity latency

This measures forward-pass latency for the capacity ladder using the actual DeltaTransformer checkpoints.
The main cost proxy for v25 should use measured latency, not checkpoint size.

## Median latency by batch size

| model | params | checkpoint MB | batch | median ms | per-item ms | throughput items/s |
|---|---:|---:|---:|---:|---:|---:|
| Tiny | 100928 | 0.39 | 1 | 0.5122 | 0.512229 | 1952.3 |
| Tiny | 100928 | 0.39 | 8 | 0.5232 | 0.065397 | 15291.3 |
| Tiny | 100928 | 0.39 | 32 | 0.5127 | 0.016022 | 62414.7 |
| Tiny | 100928 | 0.39 | 128 | 0.6416 | 0.005013 | 199500.0 |
| Tiny | 100928 | 0.39 | 512 | 1.5920 | 0.003109 | 321605.6 |
| Small | 299776 | 1.15 | 1 | 0.5092 | 0.509184 | 1963.9 |
| Small | 299776 | 1.15 | 8 | 0.5213 | 0.065157 | 15347.6 |
| Small | 299776 | 1.15 | 32 | 0.5159 | 0.016120 | 62033.1 |
| Small | 299776 | 1.15 | 128 | 0.6922 | 0.005408 | 184904.6 |
| Small | 299776 | 1.15 | 512 | 1.8846 | 0.003681 | 271682.6 |
| Medium | 1041792 | 3.99 | 1 | 0.8022 | 0.802170 | 1246.6 |
| Medium | 1041792 | 3.99 | 8 | 0.8040 | 0.100495 | 9950.7 |
| Medium | 1041792 | 3.99 | 32 | 0.8004 | 0.025013 | 39979.8 |
| Medium | 1041792 | 3.99 | 128 | 1.3177 | 0.010294 | 97139.9 |
| Medium | 1041792 | 3.99 | 512 | 4.5240 | 0.008836 | 113175.1 |
| Full | 5627136 | 21.48 | 1 | 1.0795 | 1.079493 | 926.4 |
| Full | 5627136 | 21.48 | 8 | 1.0706 | 0.133822 | 7472.6 |
| Full | 5627136 | 21.48 | 32 | 1.2500 | 0.039061 | 25600.7 |
| Full | 5627136 | 21.48 | 128 | 3.4437 | 0.026904 | 37169.3 |
| Full | 5627136 | 21.48 | 512 | 12.5868 | 0.024584 | 40677.5 |

## Batch-128 relative latency

| model | median ms @128 | relative latency vs Full |
|---|---:|---:|
| Tiny | 0.6416 | 0.186312 |
| Small | 0.6922 | 0.201019 |
| Medium | 1.3177 | 0.382637 |
| Full | 3.4437 | 1.000000 |

## Interpretation

- Use batch-128 latency as the first measured compute proxy because the evaluation scripts use grouped batches.
- If measured latency differs strongly from checkpoint-size cost, v23 should be rebuilt as v25 using latency.
- Per-item latency at batch 1 is also useful for real-time deployment analysis.

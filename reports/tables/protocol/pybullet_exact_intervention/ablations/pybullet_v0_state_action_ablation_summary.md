# PyBullet exact-intervention state/action ablation summary

| mode | best val MSE | test original top-1 | test zero top-1 | test shuffle top-1 | original-zero gap | original-shuffle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.00079370 | 0.992000 | 0.200000 | 0.160000 | 0.792000 | 0.832000 |
| action_only | 0.00749112 | 0.200000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 |
| state_only | 0.00624385 | 0.200000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 |
| no_context | 0.00850333 | 0.200000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 |

## Interpretation

If `action_only` is close to `full`, PyBullet v0 is mostly an action-template benchmark.
If `full` is much better than `action_only`, the benchmark requires state-action grounding.
If `state_only` or `no_context` is high, there is leakage or an evaluation bug.

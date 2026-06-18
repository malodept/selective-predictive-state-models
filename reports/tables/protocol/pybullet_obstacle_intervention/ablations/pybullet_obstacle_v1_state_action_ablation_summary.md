# PyBullet obstacle v1 state/action ablation summary

| mode | best val MSE | test original top-1 | test zero top-1 | test shuffle top-1 | original-zero gap | original-shuffle gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 0.00123776 | 0.852000 | 0.200000 | 0.172000 | 0.652000 | 0.680000 |
| action_only | 0.00468948 | 0.208000 | 0.200000 | 0.204000 | 0.008000 | 0.004000 |
| state_only | 0.00373905 | 0.200000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 |
| no_context | 0.00505042 | 0.200000 | 0.200000 | 0.200000 | 0.000000 | 0.000000 |

## Interpretation

`full` should strongly outperform all ablations if the benchmark requires state-action grounding.
`action_only` near chance means the model cannot solve the task by an action-template rule.
`state_only` near chance means the future branch is not predictable without the intervention action.
`no_context` near chance checks for leakage or metric artifacts.

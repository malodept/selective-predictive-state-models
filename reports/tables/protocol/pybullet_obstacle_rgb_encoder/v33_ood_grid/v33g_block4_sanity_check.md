# v33G block4 horizon sanity check

This checks whether block4 H=36 and H=72 are genuinely identical or only identical at the evaluation level.

| seed | z_current diff | z_future diff | future_xy diff | action diff | horizon values |
|---:|---:|---:|---:|---:|---|
| 20 | 0 | 0.366699 | 8.34465e-07 | 0 | 36 / 72 |
| 21 | 0 | 0.251953 | 8.34465e-07 | 0 | 36 / 72 |
| 22 | 0 | 0.353516 | 8.34465e-07 | 0 | 36 / 72 |

## Interpretation
- If future states/features are identical while horizon metadata differs, blocked=4 makes movement fully constrained, so horizon becomes physically irrelevant.
- If files differ but metrics match, the equality is an evaluation-level coincidence.
- If paths or metadata are wrong, fix v33 before using it in the paper.

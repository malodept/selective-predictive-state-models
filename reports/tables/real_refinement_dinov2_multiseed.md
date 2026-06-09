# DINOv2 real selective refinement multi-seed

| seed | cheap_error | expensive_error | best_policy | best_error | best_compute | best_selected | best_utility | cheap_utility | all_expensive_utility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.3695 | 1.2455 | threshold=0.80 | 1.3242 | 1.5386 | 0.1795 | -1.3857 | -1.4095 | -1.4055 |
| 1 | 1.3725 | 1.2701 | threshold=0.65 | 1.3430 | 1.4821 | 0.1607 | -1.4023 | -1.4125 | -1.4301 |
| 2 | 1.3744 | 1.2371 | threshold=0.65 | 1.3309 | 1.5213 | 0.1738 | -1.3917 | -1.4144 | -1.3971 |

## Summary

| metric | mean | std |
| --- | ---: | ---: |
| cheap_error | 1.3721 | 0.0020 |
| expensive_error | 1.2509 | 0.0140 |
| best_error | 1.3327 | 0.0078 |
| best_compute | 1.5140 | 0.0236 |
| best_selected | 0.1713 | 0.0079 |
| best_utility | -1.3932 | 0.0068 |
| cheap_utility | -1.4121 | 0.0020 |
| all_expensive_utility | -1.4109 | 0.0140 |

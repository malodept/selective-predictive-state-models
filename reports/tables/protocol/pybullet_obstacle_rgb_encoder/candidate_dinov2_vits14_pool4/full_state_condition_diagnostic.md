# Candidate model state-condition diagnostic

- data: `outputs/counterfactual/pybullet_obstacles_rgb/pybullet_obstacle_rgb_v1_500_seed0_dinov2_vits14_pool4.npz`
- checkpoint: `outputs/counterfactual/pybullet_obstacles_rgb/candidate_dinov2_vits14_pool4/full_seed0/checkpoint.pt`
- mode: `full`
- test groups: `50`
- chance top-1: `0.200000`

| variant | top-1 | mean rank | positive margin frac | mean margin |
| --- | ---: | ---: | ---: | ---: |
| original | 0.988000 | 1.012000 | 0.988000 | 0.091668 |
| cond_state_shuffle | 0.988000 | 1.012000 | 0.988000 | 0.090412 |
| cond_state_zero | 0.560000 | 1.664000 | 0.560000 | 0.025889 |
| cond_state_group_mean | 0.996000 | 1.004000 | 0.996000 | 0.094201 |
| action_zero | 0.200000 | 3.000000 | 0.200000 | -0.061966 |
| action_shuffle | 0.160000 | 3.100000 | 0.160000 | -0.121580 |

## Interpretation

`cond_state_shuffle` and `cond_state_zero` perturb only the state given to the predictor, while keeping the residual base state fixed.
If these variants stay close to `original`, the predictor mostly ignores the state condition and uses an action-template shortcut.

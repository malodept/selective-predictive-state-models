# Scale-5k full Transformer multiseed summary

Dataset: `pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4`.

Values are mean ± sample standard deviation over seeds 0, 1, 2.

| model | best val top-1 | test original | action zero | state shuffle | state zero |
| --- | ---: | ---: | ---: | ---: | ---: |
| full Transformer | 1.000000 ± 0.000000 | 0.997667 ± 0.000577 | 0.367667 ± 0.012423 | 0.459333 ± 0.012503 | 0.420667 ± 0.013796 |

## Interpretation

This checks whether the scale-5k full Transformer result is stable across seeds.
The key criterion is that test original remains near-perfect and far above the action/state perturbation variants.

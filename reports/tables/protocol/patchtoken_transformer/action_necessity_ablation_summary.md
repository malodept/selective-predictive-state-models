# Action-necessity ablation for patch-token dynamics

All models use the same patch-token Transformer and the same DINOv2 4x4 token states. Only the action input changes.

| action input | seeds | identity error | global error | gain vs identity | relative gain | cosine mean | positive cosine frac | global alpha | pred Δ norm med | best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive pose-difference action 7D | 3 | 1.109343 ± 0.000000 | 1.065375 ± 0.002237 | 0.043968 ± 0.002237 | 3.963389 ± 0.201617% | 0.181000 ± 0.002181 | 0.997293 ± 0.000039 | 0.630000 ± 0.008660 | 32.861644 ± 0.955390 | 1.000000 ± 0.000000 |
| relative SE(3) action 6D | 3 | 1.109343 ± 0.000000 | 1.064339 ± 0.000758 | 0.045004 ± 0.000758 | 4.056782 ± 0.068284% | 0.182145 ± 0.000592 | 0.996901 ± 0.000305 | 0.646667 ± 0.012583 | 32.333935 ± 0.557149 | 1.000000 ± 0.000000 |
| no action | 3 | 1.109343 ± 0.000000 | 1.063199 ± 0.001306 | 0.046144 ± 0.001306 | 4.159567 ± 0.117733% | 0.183480 ± 0.001726 | 0.997332 ± 0.000156 | 0.638333 ± 0.002887 | 30.472492 ± 0.692153 | 1.000000 ± 0.000000 |

## Interpretation

- Mean gain with naive 7D action: `0.043968`.
- Mean gain with relative SE(3) action: `0.045004`.
- Mean gain with no action: `0.046144`.
- Under this short-horizon protocol, removing the action does not hurt performance.
- The current result therefore supports strong patch-token latent dynamics, but not yet a strong action-conditioned world-model claim.
- A stronger action-conditioned protocol should introduce counterfactual or ambiguous futures where the same current observation can lead to different next states depending on the action.

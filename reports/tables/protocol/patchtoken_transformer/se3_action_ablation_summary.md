# Action representation ablation: naive pose difference vs relative SE(3)

Both models use the same patch-token Transformer architecture and the same DINOv2 4x4 patch-token states. Only the action representation changes.

| action representation | seeds | action dim | identity error | global error | gain vs identity | relative gain | cosine mean | positive cosine frac | global alpha | pred Δ norm med |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| naive pose difference 7D | 3 | 7 | 1.109343 ± 0.000000 | 1.065375 ± 0.002237 | 0.043968 ± 0.002237 | 3.963389 ± 0.201617% | 0.181000 ± 0.002181 | 0.997293 ± 0.000039 | 0.630000 ± 0.008660 | 32.861644 ± 0.955390 |
| relative SE(3) 6D | 3 | 6 | 1.109343 ± 0.000000 | 1.064339 ± 0.000758 | 0.045004 ± 0.000758 | 4.056782 ± 0.068284% | 0.182145 ± 0.000592 | 0.996901 ± 0.000305 | 0.646667 ± 0.012583 | 32.333935 ± 0.557149 |

## Interpretation

- Mean naive-action gain: `0.043968`.
- Mean SE(3)-action gain: `0.045004`.
- Relative SE(3) is geometrically cleaner and slightly improves the multiseed mean, but the effect is small.
- This suggests that the current model is not yet strongly exploiting the full geometry of camera motion.

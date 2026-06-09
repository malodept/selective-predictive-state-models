# Cheap/expensive gap ablation, DINOv2, multi-seed

| run | cheap error | expensive error | gap | best policy mode | best error | best compute | selected | utility | policies |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| cheap5/exp80 | 1.5143 ± 0.0098 | 1.2472 ± 0.0102 | 0.2672 ± 0.0200 | all-expensive | 1.2472 ± 0.0102 | 4.0000 ± 0.0000 | 1.0000 ± 0.0000 | -1.4072 ± 0.0102 | all-expensive, all-expensive, all-expensive |
| cheap10/exp80 | 1.3721 ± 0.0020 | 1.2509 ± 0.0140 | 0.1212 ± 0.0144 | threshold=0.65 | 1.3327 ± 0.0078 | 1.5140 ± 0.0236 | 0.1713 ± 0.0079 | -1.3932 ± 0.0068 | threshold=0.80, threshold=0.65, threshold=0.65 |
| cheap20/exp80 | 1.2967 ± 0.0040 | 1.2442 ± 0.0153 | 0.0526 ± 0.0165 | cheap-only | 1.2922 ± 0.0033 | 1.0933 ± 0.1319 | 0.0311 ± 0.0440 | -1.3359 ± 0.0029 | cheap-only, threshold=0.90, cheap-only |
| cheap10/exp40 | 1.3721 ± 0.0020 | 1.1470 ± 0.0030 | 0.2251 ± 0.0027 | all-expensive | 1.1470 ± 0.0030 | 4.0000 ± 0.0000 | 1.0000 ± 0.0000 | -1.3070 ± 0.0030 | all-expensive, all-expensive, all-expensive |
| cheap10/exp120 | 1.3721 ± 0.0020 | 1.3920 ± 0.0080 | -0.0199 ± 0.0098 | cheap-only | 1.3682 ± 0.0057 | 1.0899 ± 0.1272 | 0.0300 ± 0.0424 | -1.4118 ± 0.0020 | cheap-only, threshold=0.90, cheap-only |

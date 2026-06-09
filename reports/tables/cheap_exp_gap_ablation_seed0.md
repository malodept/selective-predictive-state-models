# Cheap/expensive gap ablation, DINOv2, seed 0

| run | cheap_error | expensive_error | gap | best_policy | best_error | best_compute | selected | utility |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| cheap5/exp80 | 1.5039 | 1.2566 | 0.2473 | all-expensive | 1.2566 | 4.0000 | 1.0000 | -1.4166 |
| cheap10/exp80 | 1.3695 | 1.2455 | 0.1240 | threshold=0.80 | 1.3242 | 1.5386 | 0.1795 | -1.3857 |
| cheap20/exp80 | 1.2922 | 1.2346 | 0.0576 | cheap-only | 1.2922 | 1.0000 | 0.0000 | -1.3322 |
| cheap10/exp40 | 1.3695 | 1.1465 | 0.2229 | all-expensive | 1.1465 | 4.0000 | 1.0000 | -1.3065 |
| cheap10/exp120 | 1.3695 | 1.3992 | -0.0298 | cheap-only | 1.3695 | 1.0000 | 0.0000 | -1.4095 |

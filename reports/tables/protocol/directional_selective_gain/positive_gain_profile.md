# Positive-gain profile for directional residual

This diagnostic compares samples where the calibrated residual improves over identity (`gain > 0`) with samples where it hurts.

| group | n | gain_mean | gain_median | gap_mean | gap_median | action_norm_mean | action_norm_median | pred_norm_mean | pred_norm_median | true_delta_norm_mean | true_delta_norm_median | cosine_mean | cosine_median | identity_error_mean | residual_error_mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | 50980 | 0.017418 | -0.001090 | 2.996469 | 3.000000 | 0.750778 | 0.574286 | 16.273375 | 15.848440 | 21.231020 | 16.493105 | 0.087518 | 0.079267 | 1.606977 | 1.589559 |
| gain>0 | 24380 | 0.051815 | 0.021826 | 3.224446 | 3.000000 | 0.859989 | 0.733212 | 16.399345 | 15.989115 | 28.506392 | 22.680826 | 0.156451 | 0.141133 | 2.690001 | 2.638186 |
| gain<=0 | 26600 | -0.014108 | -0.011640 | 2.787519 | 3.000000 | 0.650681 | 0.456507 | 16.157915 | 15.717716 | 14.562840 | 13.712872 | 0.024337 | 0.029919 | 0.614340 | 0.628448 |

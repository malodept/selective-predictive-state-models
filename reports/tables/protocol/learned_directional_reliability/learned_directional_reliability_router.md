# Learned directional reliability router

Residual alpha selected on validation: `0.164733`.

The router is trained on train, thresholded on validation, and evaluated on the held-out test environment.

| policy | score | direction | threshold | val gain | test gain | test error | test selected | selected mean gain | selected positive frac | val AUROC | test AUROC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_residual | none | - | 0.000000 | 0.018106 | 0.017418 | 1.589559 | 1.000000 | 0.017418 | 0.478227 | nan | nan |
| learned_router | HGB_classifier_proba | >= | 0.775585 | 0.018106 | 0.017417 | 1.589560 | 0.999843 | 0.017420 | 0.478263 | 0.593220 | 0.590919 |
| learned_router | HGB_gain_regressor | >= | 0.023041 | 0.018106 | 0.017418 | 1.589559 | 1.000000 | 0.017418 | 0.478227 | 0.483855 | 0.550830 |
| learned_router | logistic_classifier_proba | >= | 0.872227 | 0.018106 | 0.017418 | 1.589559 | 1.000000 | 0.017418 | 0.478227 | 0.580022 | 0.592161 |
| learned_router | ridge_gain_regressor | >= | -0.000130 | 0.018106 | 0.017418 | 1.589559 | 1.000000 | 0.017418 | 0.478227 | 0.443971 | 0.523304 |
| test_oracle_upper_bound | gain | >0 | 0.000000 | nan | 0.024779 | 1.582197 | 0.478227 | 0.051815 | 1.000000 | nan | 1.000000 |

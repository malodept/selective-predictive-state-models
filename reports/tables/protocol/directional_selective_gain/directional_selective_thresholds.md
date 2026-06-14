# Directional selective residual thresholding

Residual alpha is selected on validation: `0.164733`.

Policies choose whether to use `z_current + alpha * delta_hat` or fall back to identity.

| policy | signal | direction | threshold | val gain | test gain | test error | test selected | selected mean gain | selected positive frac |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| always_residual | none | - | 0.000000 | 0.018106 | 0.017418 | 1.589559 | 1.000000 | 0.017418 | 0.478227 |
| val_selected_threshold | gap | >= | 1.000000 | 0.018106 | 0.017418 | 1.589559 | 1.000000 | 0.017418 | 0.478227 |
| val_selected_threshold | action_norm | >= | 0.061737 | 0.018346 | 0.016816 | 1.590160 | 0.947489 | 0.017748 | 0.487216 |
| val_selected_threshold | pred_norm | >= | 5.605935 | 0.018106 | 0.017418 | 1.589559 | 1.000000 | 0.017418 | 0.478227 |
| test_oracle_upper_bound | gain | >0 | 0.000000 | nan | 0.024779 | 1.582197 | 0.478227 | 0.051815 | 1.000000 |

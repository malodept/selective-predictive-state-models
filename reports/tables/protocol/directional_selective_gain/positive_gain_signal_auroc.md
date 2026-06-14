# Deployable and oracle signals for positive-gain detection

Label: `gain > 0`, where `gain = error(identity) - error(calibrated residual)`.

| signal | availability | AUROC | best-direction AUROC | best direction |
| --- | --- | ---: | ---: | --- |
| gap | deployable | 0.587399 | 0.587399 | higher => positive gain |
| action_norm | deployable | 0.605144 | 0.605144 | higher => positive gain |
| pred_norm | deployable | 0.519731 | 0.519731 | higher => positive gain |
| true_delta_norm | oracle-only | 0.830967 | 0.830967 | higher => positive gain |
| cosine | oracle-only | 0.970528 | 0.970528 | higher => positive gain |
| identity_error | oracle-only | 0.830967 | 0.830967 | higher => positive gain |

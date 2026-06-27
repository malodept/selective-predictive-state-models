# SPSM v16 bootstrap validation of locked VoC routing

This validates v15 with paired bootstrap confidence intervals over test queries.
Positive DELTA intervals whose lower bound is above zero indicate a stable improvement.

## Main methods on 3-block, H=72

| lambda | method | utility | 95% CI | route rate |
|---:|---|---:|---:|---:|
| 0.02 | `cheap` | 0.731235 | [0.687651, 0.772397] | 0.000000 |
| 0.02 | `full` | 0.798402 | [0.762082, 0.834722] | 1.000000 |
| 0.02 | `confidence` | 0.775932 | [0.736609, 0.814628] | 0.186441 |
| 0.02 | `context_cls` | 0.769927 | [0.729103, 0.810372] | 0.123487 |
| 0.02 | `knn_context` | 0.806538 | [0.769197, 0.842665] | 0.472155 |
| 0.02 | `rh_context` | 0.801840 | [0.764794, 0.840438] | 0.464891 |
| 0.02 | `knn_context_latent` | 0.805763 | [0.767022, 0.844070] | 0.268765 |
| 0.02 | `rh_context_latent` | 0.796368 | [0.755541, 0.833028] | 0.254237 |
| 0.02 | `oracle` | 0.892591 | [0.861257, 0.919275] | 0.164649 |
| 0.05 | `cheap` | 0.731235 | [0.687651, 0.772458] | 0.000000 |
| 0.05 | `full` | 0.768402 | [0.729661, 0.804722] | 1.000000 |
| 0.05 | `confidence` | 0.770339 | [0.729056, 0.809204] | 0.186441 |
| 0.05 | `context_cls` | 0.766223 | [0.723956, 0.805085] | 0.123487 |
| 0.05 | `knn_context` | 0.779661 | [0.737530, 0.818892] | 0.338983 |
| 0.05 | `rh_context` | 0.787893 | [0.750847, 0.826758] | 0.464891 |
| 0.05 | `knn_context_latent` | 0.797700 | [0.759437, 0.834625] | 0.268765 |
| 0.05 | `rh_context_latent` | 0.793826 | [0.754722, 0.831486] | 0.249395 |
| 0.05 | `oracle` | 0.887651 | [0.857022, 0.913683] | 0.164649 |
| 0.10 | `cheap` | 0.731235 | [0.690073, 0.772397] | 0.000000 |
| 0.10 | `full` | 0.718402 | [0.679661, 0.754722] | 1.000000 |
| 0.10 | `confidence` | 0.756174 | [0.714044, 0.797343] | 0.138015 |
| 0.10 | `context_cls` | 0.754722 | [0.713311, 0.794673] | 0.104116 |
| 0.10 | `knn_context` | 0.762712 | [0.723729, 0.801465] | 0.338983 |
| 0.10 | `rh_context` | 0.760291 | [0.718880, 0.799286] | 0.314770 |
| 0.10 | `knn_context_latent` | 0.780387 | [0.739467, 0.818408] | 0.162228 |
| 0.10 | `rh_context_latent` | 0.781356 | [0.743341, 0.816949] | 0.249395 |
| 0.10 | `oracle` | 0.879419 | [0.849153, 0.907990] | 0.164649 |
| 0.20 | `cheap` | 0.731235 | [0.687651, 0.774818] | 0.000000 |
| 0.20 | `full` | 0.618402 | [0.579661, 0.654722] | 1.000000 |
| 0.20 | `confidence` | 0.731235 | [0.687651, 0.774818] | 0.000000 |
| 0.20 | `context_cls` | 0.744310 | [0.703148, 0.786441] | 0.104116 |
| 0.20 | `knn_context` | 0.728814 | [0.687167, 0.769007] | 0.338983 |
| 0.20 | `rh_context` | 0.727845 | [0.683777, 0.769007] | 0.162228 |
| 0.20 | `knn_context_latent` | 0.745763 | [0.703148, 0.785956] | 0.084746 |
| 0.20 | `rh_context_latent` | 0.734140 | [0.690073, 0.776755] | 0.058111 |
| 0.20 | `oracle` | 0.862954 | [0.832930, 0.891525] | 0.164649 |
| 0.30 | `cheap` | 0.731235 | [0.690073, 0.772397] | 0.000000 |
| 0.30 | `full` | 0.518402 | [0.479661, 0.554722] | 1.000000 |
| 0.30 | `confidence` | 0.731235 | [0.687651, 0.774818] | 0.000000 |
| 0.30 | `context_cls` | 0.727845 | [0.682082, 0.770708] | 0.092010 |
| 0.30 | `knn_context` | 0.724939 | [0.681598, 0.768765] | 0.150121 |
| 0.30 | `rh_context` | 0.712591 | [0.667791, 0.756665] | 0.142857 |
| 0.30 | `knn_context_latent` | 0.737288 | [0.693220, 0.778705] | 0.084746 |
| 0.30 | `rh_context_latent` | 0.732930 | [0.689346, 0.774818] | 0.002421 |
| 0.30 | `oracle` | 0.846489 | [0.816465, 0.876029] | 0.164649 |

## Paired latent improvement deltas on 3-block, H=72

| lambda | comparison | delta | 95% CI | stable positive? |
|---:|---|---:|---:|---|
| 0.05 | `knn_context_latent-knn_context` | 0.018039 | [-0.008841, 0.045400] | no |
| 0.05 | `rh_context_latent-rh_context` | 0.005932 | [-0.021553, 0.035596] | no |
| 0.05 | `knn_context_latent-confidence` | 0.027361 | [-0.000366, 0.055569] | no |
| 0.05 | `rh_context_latent-confidence` | 0.023487 | [-0.000972, 0.049398] | no |
| 0.05 | `knn_context_latent-context_cls` | 0.031477 | [0.002657, 0.060657] | yes |
| 0.05 | `rh_context_latent-context_cls` | 0.027603 | [-0.002788, 0.058599] | no |
| 0.10 | `knn_context_latent-knn_context` | 0.017676 | [-0.010169, 0.046737] | no |
| 0.10 | `rh_context_latent-rh_context` | 0.021065 | [-0.004600, 0.048668] | no |
| 0.10 | `knn_context_latent-confidence` | 0.024213 | [-0.002179, 0.049395] | no |
| 0.10 | `rh_context_latent-confidence` | 0.025182 | [0.000726, 0.050127] | yes |
| 0.10 | `knn_context_latent-context_cls` | 0.025666 | [-0.001211, 0.052300] | no |
| 0.10 | `rh_context_latent-context_cls` | 0.026634 | [-0.000969, 0.054237] | no |
| 0.20 | `knn_context_latent-knn_context` | 0.016949 | [-0.015012, 0.048426] | no |
| 0.20 | `rh_context_latent-rh_context` | 0.006295 | [-0.017918, 0.029056] | no |
| 0.20 | `knn_context_latent-confidence` | 0.014528 | [-0.001465, 0.031477] | no |
| 0.20 | `rh_context_latent-confidence` | 0.002906 | [-0.011138, 0.016465] | no |
| 0.20 | `knn_context_latent-context_cls` | 0.001453 | [-0.021308, 0.024697] | no |
| 0.20 | `rh_context_latent-context_cls` | -0.010169 | [-0.030993, 0.011138] | no |
| 0.30 | `knn_context_latent-knn_context` | 0.012349 | [-0.012113, 0.036320] | no |
| 0.30 | `rh_context_latent-rh_context` | 0.020339 | [-0.001943, 0.044068] | no |
| 0.30 | `knn_context_latent-confidence` | 0.006053 | [-0.010654, 0.022276] | no |
| 0.30 | `rh_context_latent-confidence` | 0.001695 | [0.000000, 0.005085] | no |
| 0.30 | `knn_context_latent-context_cls` | 0.009443 | [-0.010902, 0.030993] | no |
| 0.30 | `rh_context_latent-context_cls` | 0.005085 | [-0.013075, 0.023729] | no |

## Interpretation

- If v15 survives this bootstrap, it becomes a defensible central result.
- If intervals cross zero, we keep the result but describe it as promising rather than conclusive.
- The next step after v16 is multi-seed small-full training, not more router variants.

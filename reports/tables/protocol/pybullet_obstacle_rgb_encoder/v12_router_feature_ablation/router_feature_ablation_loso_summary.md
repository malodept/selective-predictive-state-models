# SPSM v12 router feature-source ablation — leave-one-hard-out

This experiment asks where the value-of-computation signal comes from.
It compares uncertainty-only, margin, action, context, mined-geometry-only, and context+geometry feature sets.
Each feature set is evaluated with both KNN expected-gain routing and local rescue-harm routing.

## Best method per heldout and lambda

| lambda | heldout | cheap | full | oracle | conf | context-cls | best learned | best util | best feature set | best router |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0.02 | 3-block, H=36 | 0.897311 | 0.808851 | 0.940440 | 0.852567 | 0.886944 | 0.897066 | `mined_geometry_only` | `rescue_harm` |
| 0.05 | 3-block, H=36 | 0.897311 | 0.778851 | 0.939120 | 0.844132 | 0.886064 | 0.880318 | `mined_geometry_only` | `rescue_harm` |
| 0.10 | 3-block, H=36 | 0.897311 | 0.728851 | 0.936919 | 0.868949 | 0.884597 | 0.897311 | `mined_geometry_only` | `rescue_harm` |
| 0.20 | 3-block, H=36 | 0.897311 | 0.628851 | 0.932518 | 0.855257 | 0.883619 | 0.897311 | `mined_geometry_only` | `knn_gain` |
| 0.30 | 3-block, H=36 | 0.897311 | 0.528851 | 0.928117 | 0.887775 | 0.881663 | 0.897311 | `uncertainty` | `rescue_harm` |
| 0.50 | 3-block, H=36 | 0.897311 | 0.328851 | 0.919315 | 0.897311 | 0.883863 | 0.897311 | `uncertainty` | `rescue_harm` |
| 0.02 | 3-block, H=48 | 0.853234 | 0.820796 | 0.914179 | 0.862935 | 0.861891 | 0.870945 | `context` | `rescue_harm` |
| 0.05 | 3-block, H=48 | 0.853234 | 0.790796 | 0.912313 | 0.868159 | 0.859950 | 0.866418 | `context` | `rescue_harm` |
| 0.10 | 3-block, H=48 | 0.853234 | 0.740796 | 0.909204 | 0.861940 | 0.856716 | 0.861443 | `uncertainty` | `rescue_harm` |
| 0.20 | 3-block, H=48 | 0.853234 | 0.640796 | 0.902985 | 0.849254 | 0.850249 | 0.855224 | `mined_geometry_only` | `knn_gain` |
| 0.30 | 3-block, H=48 | 0.853234 | 0.540796 | 0.896766 | 0.853234 | 0.845771 | 0.853483 | `context` | `knn_gain` |
| 0.50 | 3-block, H=48 | 0.853234 | 0.340796 | 0.884328 | 0.853234 | 0.839552 | 0.853234 | `margin` | `rescue_harm` |
| 0.02 | 3-block, H=72 | 0.731235 | 0.798402 | 0.892591 | 0.775932 | 0.769927 | 0.806538 | `context` | `knn_gain` |
| 0.05 | 3-block, H=72 | 0.731235 | 0.768402 | 0.887651 | 0.770339 | 0.766223 | 0.787893 | `context` | `rescue_harm` |
| 0.10 | 3-block, H=72 | 0.731235 | 0.718402 | 0.879419 | 0.756174 | 0.754722 | 0.762712 | `context` | `knn_gain` |
| 0.20 | 3-block, H=72 | 0.731235 | 0.618402 | 0.862954 | 0.731235 | 0.744310 | 0.744310 | `margin_action` | `knn_gain` |
| 0.30 | 3-block, H=72 | 0.731235 | 0.518402 | 0.846489 | 0.731235 | 0.727845 | 0.731235 | `uncertainty` | `rescue_harm` |
| 0.50 | 3-block, H=72 | 0.731235 | 0.318402 | 0.813559 | 0.731235 | 0.709443 | 0.732446 | `context` | `knn_gain` |

## Full learned-router table

| lambda | heldout | router | feature set | util | route | k | rule |
|---:|---|---|---|---:|---:|---:|---|
| 0.02 | 3-block, H=36 | `knn_gain` | `context` | 0.856186 | 0.222494 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=36 | `knn_gain` | `context_geometry` | 0.838973 | 0.227384 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=36 | `knn_gain` | `margin` | 0.859853 | 0.283619 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=36 | `knn_gain` | `margin_action` | 0.858044 | 0.374083 | 25 | `gain_threshold` |
| 0.02 | 3-block, H=36 | `knn_gain` | `mined_geometry_only` | 0.866210 | 0.210269 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=36 | `knn_gain` | `uncertainty` | 0.842200 | 0.310513 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=36 | `rescue_harm` | `context` | 0.865086 | 0.266504 | 100 | `thr=0.0200,cap=0.2000` |
| 0.02 | 3-block, H=36 | `rescue_harm` | `context_geometry` | 0.847726 | 0.278729 | 25 | `thr=0.0200,cap=0.3500` |
| 0.02 | 3-block, H=36 | `rescue_harm` | `margin` | 0.871100 | 0.210269 | 25 | `thr=0.0800,cap=0.2800` |
| 0.02 | 3-block, H=36 | `rescue_harm` | `margin_action` | 0.849584 | 0.430318 | 100 | `thr=0.0200,cap=0.2200` |
| 0.02 | 3-block, H=36 | `rescue_harm` | `mined_geometry_only` | 0.897066 | 0.012225 | 200 | `thr=0.0300,cap=0.1250` |
| 0.02 | 3-block, H=36 | `rescue_harm` | `uncertainty` | 0.855550 | 0.254279 | 25 | `thr=0.0400,cap=0.2000` |
| 0.05 | 3-block, H=36 | `knn_gain` | `context` | 0.849511 | 0.222494 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=36 | `knn_gain` | `context_geometry` | 0.832152 | 0.227384 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=36 | `knn_gain` | `margin` | 0.858924 | 0.229829 | 25 | `gain_threshold` |
| 0.05 | 3-block, H=36 | `knn_gain` | `margin_action` | 0.846822 | 0.374083 | 25 | `tuned_threshold` |
| 0.05 | 3-block, H=36 | `knn_gain` | `mined_geometry_only` | 0.859902 | 0.210269 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=36 | `knn_gain` | `uncertainty` | 0.832885 | 0.310513 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=36 | `rescue_harm` | `context` | 0.861247 | 0.232274 | 100 | `thr=0.0300,cap=0.2000` |
| 0.05 | 3-block, H=36 | `rescue_harm` | `context_geometry` | 0.839364 | 0.278729 | 25 | `thr=0.0275,cap=0.3500` |
| 0.05 | 3-block, H=36 | `rescue_harm` | `margin` | 0.864792 | 0.210269 | 25 | `thr=0.0800,cap=0.2800` |
| 0.05 | 3-block, H=36 | `rescue_harm` | `margin_action` | 0.860513 | 0.246944 | 50 | `thr=0.0800,cap=0.2000` |
| 0.05 | 3-block, H=36 | `rescue_harm` | `mined_geometry_only` | 0.880318 | 0.095355 | 15 | `thr=0.0667,cap=0.1333` |
| 0.05 | 3-block, H=36 | `rescue_harm` | `uncertainty` | 0.864670 | 0.163814 | 50 | `thr=0.1000,cap=0.2000` |
| 0.10 | 3-block, H=36 | `knn_gain` | `context` | 0.867971 | 0.146699 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=36 | `knn_gain` | `context_geometry` | 0.878973 | 0.061125 | 15 | `tuned_threshold` |
| 0.10 | 3-block, H=36 | `knn_gain` | `margin` | 0.847433 | 0.229829 | 25 | `tuned_threshold` |
| 0.10 | 3-block, H=36 | `knn_gain` | `margin_action` | 0.857946 | 0.149144 | 15 | `tuned_threshold` |
| 0.10 | 3-block, H=36 | `knn_gain` | `mined_geometry_only` | 0.896822 | 0.004890 | 15 | `tuned_threshold` |
| 0.10 | 3-block, H=36 | `knn_gain` | `uncertainty` | 0.844254 | 0.212714 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=36 | `rescue_harm` | `context` | 0.884108 | 0.107579 | 15 | `thr=0.1333,cap=0.1333` |
| 0.10 | 3-block, H=36 | `rescue_harm` | `context_geometry` | 0.883863 | 0.061125 | 50 | `thr=0.1200,cap=0.3400` |
| 0.10 | 3-block, H=36 | `rescue_harm` | `margin` | 0.867482 | 0.127139 | 200 | `thr=0.0900,cap=0.1050` |
| 0.10 | 3-block, H=36 | `rescue_harm` | `margin_action` | 0.848166 | 0.246944 | 50 | `thr=0.0800,cap=0.2000` |
| 0.10 | 3-block, H=36 | `rescue_harm` | `mined_geometry_only` | 0.897311 | 0.000000 | 15 | `thr=0.3333,cap=0.0667` |
| 0.10 | 3-block, H=36 | `rescue_harm` | `uncertainty` | 0.877506 | 0.124694 | 50 | `thr=0.1800,cap=0.1200` |
| 0.20 | 3-block, H=36 | `knn_gain` | `context` | 0.880196 | 0.085575 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=36 | `knn_gain` | `context_geometry` | 0.872861 | 0.061125 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=36 | `knn_gain` | `margin` | 0.866504 | 0.105134 | 15 | `gain_threshold` |
| 0.20 | 3-block, H=36 | `knn_gain` | `margin_action` | 0.843032 | 0.149144 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=36 | `knn_gain` | `mined_geometry_only` | 0.897311 | 0.000000 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=36 | `knn_gain` | `uncertainty` | 0.856724 | 0.141809 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=36 | `rescue_harm` | `context` | 0.877751 | 0.036675 | 15 | `thr=0.2667,cap=0.1333` |
| 0.20 | 3-block, H=36 | `rescue_harm` | `context_geometry` | 0.890954 | 0.044010 | 50 | `thr=0.1400,cap=0.2000` |
| 0.20 | 3-block, H=36 | `rescue_harm` | `margin` | 0.858191 | 0.122249 | 200 | `thr=0.1300,cap=0.1050` |
| 0.20 | 3-block, H=36 | `rescue_harm` | `margin_action` | 0.863570 | 0.107579 | 50 | `thr=0.1800,cap=0.1000` |
| 0.20 | 3-block, H=36 | `rescue_harm` | `mined_geometry_only` | 0.897311 | 0.000000 | 15 | `thr=0.2667,cap=0.0000` |
| 0.20 | 3-block, H=36 | `rescue_harm` | `uncertainty` | 0.865037 | 0.124694 | 50 | `thr=0.1800,cap=0.1200` |
| 0.30 | 3-block, H=36 | `knn_gain` | `context` | 0.887286 | 0.017115 | 15 | `gain_threshold` |
| 0.30 | 3-block, H=36 | `knn_gain` | `context_geometry` | 0.892665 | 0.007335 | 15 | `gain_threshold` |
| 0.30 | 3-block, H=36 | `knn_gain` | `margin` | 0.876528 | 0.036675 | 50 | `gain_threshold` |
| 0.30 | 3-block, H=36 | `knn_gain` | `margin_action` | 0.866993 | 0.068460 | 50 | `tuned_threshold` |
| 0.30 | 3-block, H=36 | `knn_gain` | `mined_geometry_only` | 0.897311 | 0.000000 | 15 | `gain_threshold` |
| 0.30 | 3-block, H=36 | `knn_gain` | `uncertainty` | 0.886553 | 0.019560 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=36 | `rescue_harm` | `context` | 0.883374 | 0.022005 | 100 | `thr=0.1442,cap=0.0700` |
| 0.30 | 3-block, H=36 | `rescue_harm` | `context_geometry` | 0.897311 | 0.000000 | 200 | `thr=0.0550,cap=0.0150` |
| 0.30 | 3-block, H=36 | `rescue_harm` | `margin` | 0.869193 | 0.061125 | 200 | `thr=0.2000,cap=0.1050` |
| 0.30 | 3-block, H=36 | `rescue_harm` | `margin_action` | 0.885575 | 0.014670 | 100 | `thr=0.1500,cap=0.0500` |
| 0.30 | 3-block, H=36 | `rescue_harm` | `mined_geometry_only` | 0.897311 | 0.000000 | 15 | `thr=0.2667,cap=0.0000` |
| 0.30 | 3-block, H=36 | `rescue_harm` | `uncertainty` | 0.897311 | 0.000000 | 25 | `thr=0.4000,cap=0.0400` |
| 0.50 | 3-block, H=36 | `knn_gain` | `context` | 0.897311 | 0.000000 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=36 | `knn_gain` | `context_geometry` | 0.896088 | 0.002445 | 15 | `tuned_threshold` |
| 0.50 | 3-block, H=36 | `knn_gain` | `margin` | 0.897311 | 0.000000 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=36 | `knn_gain` | `margin_action` | 0.897311 | 0.004890 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=36 | `knn_gain` | `mined_geometry_only` | 0.897311 | 0.000000 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=36 | `knn_gain` | `uncertainty` | 0.882641 | 0.019560 | 15 | `tuned_threshold` |
| 0.50 | 3-block, H=36 | `rescue_harm` | `context` | 0.897311 | 0.000000 | 25 | `thr=0.5000,cap=0.0000` |
| 0.50 | 3-block, H=36 | `rescue_harm` | `context_geometry` | 0.897311 | 0.000000 | 200 | `thr=0.0550,cap=0.0050` |
| 0.50 | 3-block, H=36 | `rescue_harm` | `margin` | 0.880196 | 0.014670 | 100 | `thr=0.2900,cap=0.0500` |
| 0.50 | 3-block, H=36 | `rescue_harm` | `margin_action` | 0.896088 | 0.002445 | 200 | `thr=0.0450,cap=0.0050` |
| 0.50 | 3-block, H=36 | `rescue_harm` | `mined_geometry_only` | 0.897311 | 0.000000 | 15 | `thr=0.5000,cap=0.0000` |
| 0.50 | 3-block, H=36 | `rescue_harm` | `uncertainty` | 0.897311 | 0.000000 | 15 | `thr=0.5000,cap=0.0000` |
| 0.02 | 3-block, H=48 | `knn_gain` | `context` | 0.866219 | 0.097015 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=48 | `knn_gain` | `context_geometry` | 0.854677 | 0.052239 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=48 | `knn_gain` | `margin` | 0.846716 | 0.201493 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=48 | `knn_gain` | `margin_action` | 0.851592 | 0.330846 | 25 | `gain_threshold` |
| 0.02 | 3-block, H=48 | `knn_gain` | `mined_geometry_only` | 0.848507 | 0.111940 | 25 | `gain_threshold` |
| 0.02 | 3-block, H=48 | `knn_gain` | `uncertainty` | 0.842189 | 0.303483 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=48 | `rescue_harm` | `context` | 0.870945 | 0.109453 | 25 | `thr=0.0400,cap=0.2000` |
| 0.02 | 3-block, H=48 | `rescue_harm` | `context_geometry` | 0.852388 | 0.042289 | 25 | `thr=0.0400,cap=0.2400` |
| 0.02 | 3-block, H=48 | `rescue_harm` | `margin` | 0.853035 | 0.258706 | 25 | `thr=0.0200,cap=0.2000` |
| 0.02 | 3-block, H=48 | `rescue_harm` | `margin_action` | 0.855721 | 0.373134 | 50 | `thr=0.0200,cap=0.3200` |
| 0.02 | 3-block, H=48 | `rescue_harm` | `mined_geometry_only` | 0.855622 | 0.004975 | 100 | `thr=0.0400,cap=0.1500` |
| 0.02 | 3-block, H=48 | `rescue_harm` | `uncertainty` | 0.859552 | 0.181592 | 200 | `thr=0.0250,cap=0.1600` |
| 0.05 | 3-block, H=48 | `knn_gain` | `context` | 0.863308 | 0.097015 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=48 | `knn_gain` | `context_geometry` | 0.847886 | 0.007463 | 15 | `tuned_threshold` |
| 0.05 | 3-block, H=48 | `knn_gain` | `margin` | 0.840672 | 0.201493 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=48 | `knn_gain` | `margin_action` | 0.845025 | 0.263682 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=48 | `knn_gain` | `mined_geometry_only` | 0.843159 | 0.101990 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=48 | `knn_gain` | `uncertainty` | 0.833085 | 0.303483 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=48 | `rescue_harm` | `context` | 0.866418 | 0.084577 | 25 | `thr=0.0400,cap=0.2000` |
| 0.05 | 3-block, H=48 | `rescue_harm` | `context_geometry` | 0.851119 | 0.042289 | 25 | `thr=0.0400,cap=0.2400` |
| 0.05 | 3-block, H=48 | `rescue_harm` | `margin` | 0.862935 | 0.104478 | 50 | `thr=0.1200,cap=0.2000` |
| 0.05 | 3-block, H=48 | `rescue_harm` | `margin_action` | 0.858333 | 0.097015 | 100 | `thr=0.0600,cap=0.2600` |
| 0.05 | 3-block, H=48 | `rescue_harm` | `mined_geometry_only` | 0.855473 | 0.004975 | 100 | `thr=0.0400,cap=0.1500` |
| 0.05 | 3-block, H=48 | `rescue_harm` | `uncertainty` | 0.866045 | 0.092040 | 200 | `thr=0.0500,cap=0.1600` |
| 0.10 | 3-block, H=48 | `knn_gain` | `context` | 0.851244 | 0.044776 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=48 | `knn_gain` | `context_geometry` | 0.850498 | 0.002488 | 15 | `tuned_threshold` |
| 0.10 | 3-block, H=48 | `knn_gain` | `margin` | 0.830597 | 0.201493 | 15 | `tuned_threshold` |
| 0.10 | 3-block, H=48 | `knn_gain` | `margin_action` | 0.845025 | 0.156716 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=48 | `knn_gain` | `mined_geometry_only` | 0.855473 | 0.002488 | 15 | `tuned_threshold` |
| 0.10 | 3-block, H=48 | `knn_gain` | `uncertainty` | 0.842040 | 0.111940 | 25 | `gain_threshold` |
| 0.10 | 3-block, H=48 | `rescue_harm` | `context` | 0.850746 | 0.024876 | 15 | `thr=0.1333,cap=0.1333` |
| 0.10 | 3-block, H=48 | `rescue_harm` | `context_geometry` | 0.847512 | 0.007463 | 15 | `thr=0.1000,cap=0.3500` |
| 0.10 | 3-block, H=48 | `rescue_harm` | `margin` | 0.857711 | 0.104478 | 50 | `thr=0.1200,cap=0.2000` |
| 0.10 | 3-block, H=48 | `rescue_harm` | `margin_action` | 0.848259 | 0.074627 | 100 | `thr=0.0700,cap=0.2000` |
| 0.10 | 3-block, H=48 | `rescue_harm` | `mined_geometry_only` | 0.853234 | 0.000000 | 15 | `thr=0.3333,cap=0.0667` |
| 0.10 | 3-block, H=48 | `rescue_harm` | `uncertainty` | 0.861443 | 0.092040 | 200 | `thr=0.0450,cap=0.1600` |
| 0.20 | 3-block, H=48 | `knn_gain` | `context` | 0.854229 | 0.007463 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=48 | `knn_gain` | `context_geometry` | 0.850249 | 0.002488 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=48 | `knn_gain` | `margin` | 0.843284 | 0.049751 | 15 | `gain_threshold` |
| 0.20 | 3-block, H=48 | `knn_gain` | `margin_action` | 0.844776 | 0.042289 | 15 | `gain_threshold` |
| 0.20 | 3-block, H=48 | `knn_gain` | `mined_geometry_only` | 0.855224 | 0.002488 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=48 | `knn_gain` | `uncertainty` | 0.848259 | 0.037313 | 50 | `gain_threshold` |
| 0.20 | 3-block, H=48 | `rescue_harm` | `context` | 0.853731 | 0.009950 | 15 | `thr=0.2667,cap=0.1333` |
| 0.20 | 3-block, H=48 | `rescue_harm` | `context_geometry` | 0.853234 | 0.000000 | 200 | `thr=0.0667,cap=0.2000` |
| 0.20 | 3-block, H=48 | `rescue_harm` | `margin` | 0.844776 | 0.017413 | 25 | `thr=0.2800,cap=0.0400` |
| 0.20 | 3-block, H=48 | `rescue_harm` | `margin_action` | 0.850746 | 0.012438 | 25 | `thr=0.2000,cap=0.0400` |
| 0.20 | 3-block, H=48 | `rescue_harm` | `mined_geometry_only` | 0.853234 | 0.000000 | 15 | `thr=0.2667,cap=0.0000` |
| 0.20 | 3-block, H=48 | `rescue_harm` | `uncertainty` | 0.852239 | 0.017413 | 25 | `thr=0.2000,cap=0.0000` |
| 0.30 | 3-block, H=48 | `knn_gain` | `context` | 0.853483 | 0.007463 | 15 | `gain_threshold` |
| 0.30 | 3-block, H=48 | `knn_gain` | `context_geometry` | 0.853234 | 0.000000 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=48 | `knn_gain` | `margin` | 0.837562 | 0.027363 | 15 | `gain_threshold` |
| 0.30 | 3-block, H=48 | `knn_gain` | `margin_action` | 0.853234 | 0.000000 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=48 | `knn_gain` | `mined_geometry_only` | 0.853234 | 0.000000 | 15 | `gain_threshold` |
| 0.30 | 3-block, H=48 | `knn_gain` | `uncertainty` | 0.848507 | 0.007463 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=48 | `rescue_harm` | `context` | 0.851741 | 0.004975 | 25 | `thr=0.2000,cap=0.0800` |
| 0.30 | 3-block, H=48 | `rescue_harm` | `context_geometry` | 0.853234 | 0.000000 | 200 | `thr=0.0750,cap=0.2000` |
| 0.30 | 3-block, H=48 | `rescue_harm` | `margin` | 0.853234 | 0.000000 | 200 | `thr=0.0750,cap=0.0900` |
| 0.30 | 3-block, H=48 | `rescue_harm` | `margin_action` | 0.849502 | 0.012438 | 25 | `thr=0.2000,cap=0.0400` |
| 0.30 | 3-block, H=48 | `rescue_harm` | `mined_geometry_only` | 0.853234 | 0.000000 | 15 | `thr=0.2667,cap=0.0000` |
| 0.30 | 3-block, H=48 | `rescue_harm` | `uncertainty` | 0.850498 | 0.017413 | 25 | `thr=0.2000,cap=0.0000` |
| 0.50 | 3-block, H=48 | `knn_gain` | `context` | 0.853234 | 0.000000 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=48 | `knn_gain` | `context_geometry` | 0.853234 | 0.000000 | 15 | `tuned_threshold` |
| 0.50 | 3-block, H=48 | `knn_gain` | `margin` | 0.853234 | 0.000000 | 15 | `tuned_threshold` |
| 0.50 | 3-block, H=48 | `knn_gain` | `margin_action` | 0.853234 | 0.000000 | 15 | `tuned_threshold` |
| 0.50 | 3-block, H=48 | `knn_gain` | `mined_geometry_only` | 0.853234 | 0.000000 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=48 | `knn_gain` | `uncertainty` | 0.847015 | 0.007463 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=48 | `rescue_harm` | `context` | 0.853234 | 0.000000 | 50 | `thr=0.3400,cap=0.0800` |
| 0.50 | 3-block, H=48 | `rescue_harm` | `context_geometry` | 0.853234 | 0.000000 | 100 | `thr=0.1500,cap=0.0100` |
| 0.50 | 3-block, H=48 | `rescue_harm` | `margin` | 0.853234 | 0.000000 | 200 | `thr=0.0750,cap=0.0900` |
| 0.50 | 3-block, H=48 | `rescue_harm` | `margin_action` | 0.853234 | 0.000000 | 50 | `thr=0.2200,cap=0.0400` |
| 0.50 | 3-block, H=48 | `rescue_harm` | `mined_geometry_only` | 0.853234 | 0.000000 | 15 | `thr=0.5000,cap=0.0000` |
| 0.50 | 3-block, H=48 | `rescue_harm` | `uncertainty` | 0.847015 | 0.017413 | 25 | `thr=0.2000,cap=0.0000` |
| 0.02 | 3-block, H=72 | `knn_gain` | `context` | 0.806538 | 0.472155 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=72 | `knn_gain` | `context_geometry` | 0.784988 | 0.338983 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=72 | `knn_gain` | `margin` | 0.752494 | 0.268765 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=72 | `knn_gain` | `margin_action` | 0.776949 | 0.256659 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=72 | `knn_gain` | `mined_geometry_only` | 0.737240 | 0.184019 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=72 | `knn_gain` | `uncertainty` | 0.770266 | 0.227603 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=72 | `rescue_harm` | `context` | 0.801840 | 0.464891 | 15 | `thr=0.0200,cap=0.1333` |
| 0.02 | 3-block, H=72 | `rescue_harm` | `context_geometry` | 0.744407 | 0.188862 | 25 | `thr=0.0200,cap=0.1000` |
| 0.02 | 3-block, H=72 | `rescue_harm` | `margin` | 0.769201 | 0.159806 | 100 | `thr=0.0700,cap=0.2000` |
| 0.02 | 3-block, H=72 | `rescue_harm` | `margin_action` | 0.768232 | 0.208232 | 200 | `thr=0.0150,cap=0.2000` |
| 0.02 | 3-block, H=72 | `rescue_harm` | `mined_geometry_only` | 0.731235 | 0.000000 | 200 | `thr=0.0300,cap=0.0100` |
| 0.02 | 3-block, H=72 | `rescue_harm` | `uncertainty` | 0.775690 | 0.198547 | 100 | `thr=0.0200,cap=0.2000` |
| 0.05 | 3-block, H=72 | `knn_gain` | `context` | 0.779661 | 0.338983 | 15 | `tuned_threshold` |
| 0.05 | 3-block, H=72 | `knn_gain` | `context_geometry` | 0.774818 | 0.338983 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=72 | `knn_gain` | `margin` | 0.744431 | 0.268765 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=72 | `knn_gain` | `margin_action` | 0.769249 | 0.256659 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=72 | `knn_gain` | `mined_geometry_only` | 0.731719 | 0.184019 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=72 | `knn_gain` | `uncertainty` | 0.763438 | 0.227603 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=72 | `rescue_harm` | `context` | 0.787893 | 0.464891 | 15 | `thr=0.0500,cap=0.1333` |
| 0.05 | 3-block, H=72 | `rescue_harm` | `context_geometry` | 0.734988 | 0.167070 | 15 | `thr=0.0500,cap=0.0667` |
| 0.05 | 3-block, H=72 | `rescue_harm` | `margin` | 0.729661 | 0.079903 | 50 | `thr=0.0400,cap=0.1200` |
| 0.05 | 3-block, H=72 | `rescue_harm` | `margin_action` | 0.761985 | 0.208232 | 200 | `thr=0.0150,cap=0.2000` |
| 0.05 | 3-block, H=72 | `rescue_harm` | `mined_geometry_only` | 0.730872 | 0.007264 | 100 | `thr=0.0400,cap=0.0100` |
| 0.05 | 3-block, H=72 | `rescue_harm` | `uncertainty` | 0.769734 | 0.198547 | 100 | `thr=0.0200,cap=0.2000` |
| 0.10 | 3-block, H=72 | `knn_gain` | `context` | 0.762712 | 0.338983 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=72 | `knn_gain` | `context_geometry` | 0.744068 | 0.210654 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=72 | `knn_gain` | `margin` | 0.748910 | 0.162228 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=72 | `knn_gain` | `margin_action` | 0.755932 | 0.164649 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=72 | `knn_gain` | `mined_geometry_only` | 0.730993 | 0.002421 | 15 | `tuned_threshold` |
| 0.10 | 3-block, H=72 | `knn_gain` | `uncertainty` | 0.742131 | 0.084746 | 50 | `tuned_threshold` |
| 0.10 | 3-block, H=72 | `rescue_harm` | `context` | 0.760291 | 0.314770 | 100 | `thr=0.0500,cap=0.1800` |
| 0.10 | 3-block, H=72 | `rescue_harm` | `context_geometry` | 0.731961 | 0.113801 | 15 | `thr=0.1000,cap=0.0667` |
| 0.10 | 3-block, H=72 | `rescue_harm` | `margin` | 0.739952 | 0.033898 | 50 | `thr=0.1400,cap=0.1200` |
| 0.10 | 3-block, H=72 | `rescue_harm` | `margin_action` | 0.729540 | 0.041162 | 50 | `thr=0.0400,cap=0.0400` |
| 0.10 | 3-block, H=72 | `rescue_harm` | `mined_geometry_only` | 0.731235 | 0.000000 | 15 | `thr=1000000000.0000,cap=0.0000` |
| 0.10 | 3-block, H=72 | `rescue_harm` | `uncertainty` | 0.747700 | 0.077482 | 50 | `thr=0.1000,cap=0.1500` |
| 0.20 | 3-block, H=72 | `knn_gain` | `context` | 0.728814 | 0.338983 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=72 | `knn_gain` | `context_geometry` | 0.723002 | 0.210654 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=72 | `knn_gain` | `margin` | 0.743341 | 0.060533 | 50 | `tuned_threshold` |
| 0.20 | 3-block, H=72 | `knn_gain` | `margin_action` | 0.744310 | 0.104116 | 25 | `tuned_threshold` |
| 0.20 | 3-block, H=72 | `knn_gain` | `mined_geometry_only` | 0.731235 | 0.000000 | 15 | `gain_threshold` |
| 0.20 | 3-block, H=72 | `knn_gain` | `uncertainty` | 0.736077 | 0.024213 | 50 | `tuned_threshold` |
| 0.20 | 3-block, H=72 | `rescue_harm` | `context` | 0.727845 | 0.162228 | 100 | `thr=0.0900,cap=0.1800` |
| 0.20 | 3-block, H=72 | `rescue_harm` | `context_geometry` | 0.731235 | 0.000000 | 100 | `thr=0.0748,cap=0.0500` |
| 0.20 | 3-block, H=72 | `rescue_harm` | `margin` | 0.726392 | 0.012107 | 50 | `thr=0.1800,cap=0.1200` |
| 0.20 | 3-block, H=72 | `rescue_harm` | `margin_action` | 0.731235 | 0.000000 | 100 | `thr=0.1500,cap=0.1000` |
| 0.20 | 3-block, H=72 | `rescue_harm` | `mined_geometry_only` | 0.731235 | 0.000000 | 15 | `thr=1000000000.0000,cap=0.0000` |
| 0.20 | 3-block, H=72 | `rescue_harm` | `uncertainty` | 0.736077 | 0.012107 | 50 | `thr=0.2000,cap=0.0800` |
| 0.30 | 3-block, H=72 | `knn_gain` | `context` | 0.724939 | 0.150121 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=72 | `knn_gain` | `context_geometry` | 0.729782 | 0.053269 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=72 | `knn_gain` | `margin` | 0.725182 | 0.012107 | 25 | `gain_threshold` |
| 0.30 | 3-block, H=72 | `knn_gain` | `margin_action` | 0.729782 | 0.004843 | 100 | `tuned_threshold` |
| 0.30 | 3-block, H=72 | `knn_gain` | `mined_geometry_only` | 0.731235 | 0.000000 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=72 | `knn_gain` | `uncertainty` | 0.730508 | 0.002421 | 50 | `tuned_threshold` |
| 0.30 | 3-block, H=72 | `rescue_harm` | `context` | 0.712591 | 0.142857 | 100 | `thr=0.0900,cap=0.1500` |
| 0.30 | 3-block, H=72 | `rescue_harm` | `context_geometry` | 0.731235 | 0.000000 | 100 | `thr=0.0748,cap=0.0200` |
| 0.30 | 3-block, H=72 | `rescue_harm` | `margin` | 0.727361 | 0.004843 | 25 | `thr=0.3000,cap=0.0000` |
| 0.30 | 3-block, H=72 | `rescue_harm` | `margin_action` | 0.731235 | 0.000000 | 100 | `thr=0.1500,cap=0.1000` |
| 0.30 | 3-block, H=72 | `rescue_harm` | `mined_geometry_only` | 0.731235 | 0.000000 | 15 | `thr=1000000000.0000,cap=0.0000` |
| 0.30 | 3-block, H=72 | `rescue_harm` | `uncertainty` | 0.731235 | 0.000000 | 15 | `thr=0.3333,cap=0.0000` |
| 0.50 | 3-block, H=72 | `knn_gain` | `context` | 0.732446 | 0.002421 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=72 | `knn_gain` | `context_geometry` | 0.731235 | 0.000000 | 25 | `tuned_threshold` |
| 0.50 | 3-block, H=72 | `knn_gain` | `margin` | 0.728814 | 0.004843 | 25 | `tuned_threshold` |
| 0.50 | 3-block, H=72 | `knn_gain` | `margin_action` | 0.731235 | 0.000000 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=72 | `knn_gain` | `mined_geometry_only` | 0.731235 | 0.000000 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=72 | `knn_gain` | `uncertainty` | 0.730024 | 0.002421 | 50 | `tuned_threshold` |
| 0.50 | 3-block, H=72 | `rescue_harm` | `context` | 0.732446 | 0.002421 | 15 | `thr=0.5000,cap=0.0667` |
| 0.50 | 3-block, H=72 | `rescue_harm` | `context_geometry` | 0.731235 | 0.000000 | 200 | `thr=0.0750,cap=0.0050` |
| 0.50 | 3-block, H=72 | `rescue_harm` | `margin` | 0.731235 | 0.000000 | 15 | `thr=0.5000,cap=0.0000` |
| 0.50 | 3-block, H=72 | `rescue_harm` | `margin_action` | 0.731235 | 0.000000 | 100 | `thr=0.1500,cap=0.1000` |
| 0.50 | 3-block, H=72 | `rescue_harm` | `mined_geometry_only` | 0.731235 | 0.000000 | 15 | `thr=0.5000,cap=0.0000` |
| 0.50 | 3-block, H=72 | `rescue_harm` | `uncertainty` | 0.731235 | 0.000000 | 15 | `thr=0.3333,cap=0.0000` |

## Interpretation guide

- If `uncertainty` or `margin` is best, the value-of-computation signal is mostly cheap-model risk.
- If `context` beats `margin_action`, horizon/velocity add useful shift information.
- If `mined_geometry_only` or `context_geometry` wins, the current geometry features are useful.
- If rescue-harm wins over KNN gain, decomposing rescue and harm is useful.

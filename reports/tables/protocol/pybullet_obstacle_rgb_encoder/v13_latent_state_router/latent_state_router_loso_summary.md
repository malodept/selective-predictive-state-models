# SPSM v13 observable latent-state router — leave-one-hard-out

This experiment adds features computed only from the current observed latent state `z_i = phi(x_i)`.
Unlike candidate-mined geometry, these features are available before deciding whether to call the expensive model.

## Best method per heldout and lambda

| lambda | heldout | cheap | full | oracle | conf | context-cls | best learned | best feature set | best router |
|---:|---|---:|---:|---:|---:|---:|---:|---|---|
| 0.02 | 3-block, H=36 | 0.897311 | 0.808851 | 0.940440 | 0.852567 | 0.886944 | 0.877653 | `latent_only` | `knn_gain` |
| 0.05 | 3-block, H=36 | 0.897311 | 0.778851 | 0.939120 | 0.844132 | 0.886064 | 0.883252 | `context_latent` | `knn_gain` |
| 0.10 | 3-block, H=36 | 0.897311 | 0.728851 | 0.936919 | 0.868949 | 0.884597 | 0.884108 | `latent_only` | `rescue_harm` |
| 0.20 | 3-block, H=36 | 0.897311 | 0.628851 | 0.932518 | 0.855257 | 0.883619 | 0.888020 | `latent_only` | `rescue_harm` |
| 0.30 | 3-block, H=36 | 0.897311 | 0.528851 | 0.928117 | 0.887775 | 0.881663 | 0.897311 | `uncertainty` | `rescue_harm` |
| 0.50 | 3-block, H=36 | 0.897311 | 0.328851 | 0.919315 | 0.897311 | 0.883863 | 0.897311 | `uncertainty` | `rescue_harm` |
| 0.02 | 3-block, H=48 | 0.853234 | 0.820796 | 0.914179 | 0.862935 | 0.861891 | 0.870945 | `context` | `rescue_harm` |
| 0.05 | 3-block, H=48 | 0.853234 | 0.790796 | 0.912313 | 0.868159 | 0.859950 | 0.866791 | `context_latent` | `knn_gain` |
| 0.10 | 3-block, H=48 | 0.853234 | 0.740796 | 0.909204 | 0.861940 | 0.856716 | 0.861443 | `uncertainty` | `rescue_harm` |
| 0.20 | 3-block, H=48 | 0.853234 | 0.640796 | 0.902985 | 0.849254 | 0.850249 | 0.854229 | `context` | `knn_gain` |
| 0.30 | 3-block, H=48 | 0.853234 | 0.540796 | 0.896766 | 0.853234 | 0.845771 | 0.853483 | `context` | `knn_gain` |
| 0.50 | 3-block, H=48 | 0.853234 | 0.340796 | 0.884328 | 0.853234 | 0.839552 | 0.854478 | `context_latent` | `rescue_harm` |
| 0.02 | 3-block, H=72 | 0.731235 | 0.798402 | 0.892591 | 0.775932 | 0.769927 | 0.806538 | `context` | `knn_gain` |
| 0.05 | 3-block, H=72 | 0.731235 | 0.768402 | 0.887651 | 0.770339 | 0.766223 | 0.797700 | `context_latent` | `knn_gain` |
| 0.10 | 3-block, H=72 | 0.731235 | 0.718402 | 0.879419 | 0.756174 | 0.754722 | 0.781356 | `context_latent` | `rescue_harm` |
| 0.20 | 3-block, H=72 | 0.731235 | 0.618402 | 0.862954 | 0.731235 | 0.744310 | 0.745763 | `context_latent` | `knn_gain` |
| 0.30 | 3-block, H=72 | 0.731235 | 0.518402 | 0.846489 | 0.731235 | 0.727845 | 0.737288 | `context_latent` | `knn_gain` |
| 0.50 | 3-block, H=72 | 0.731235 | 0.318402 | 0.813559 | 0.731235 | 0.709443 | 0.732446 | `context` | `knn_gain` |

## Full learned-router table

| lambda | heldout | router | feature set | util | route | k | rule |
|---:|---|---|---|---:|---:|---:|---|
| 0.02 | 3-block, H=36 | `knn_gain` | `context` | 0.856186 | 0.222494 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=36 | `knn_gain` | `context_latent` | 0.875012 | 0.259169 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=36 | `knn_gain` | `latent_only` | 0.877653 | 0.371638 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=36 | `knn_gain` | `margin` | 0.859853 | 0.283619 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=36 | `knn_gain` | `uncertainty` | 0.842200 | 0.310513 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=36 | `rescue_harm` | `context` | 0.865086 | 0.266504 | 100 | `thr=0.0200,cap=0.2000` |
| 0.02 | 3-block, H=36 | `rescue_harm` | `context_latent` | 0.876773 | 0.171149 | 25 | `thr=0.0800,cap=0.1600` |
| 0.02 | 3-block, H=36 | `rescue_harm` | `latent_only` | 0.867628 | 0.506112 | 50 | `thr=0.0200,cap=0.0800` |
| 0.02 | 3-block, H=36 | `rescue_harm` | `margin` | 0.871100 | 0.210269 | 25 | `thr=0.0800,cap=0.2800` |
| 0.02 | 3-block, H=36 | `rescue_harm` | `uncertainty` | 0.855550 | 0.254279 | 25 | `thr=0.0400,cap=0.2000` |
| 0.05 | 3-block, H=36 | `knn_gain` | `context` | 0.849511 | 0.222494 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=36 | `knn_gain` | `context_latent` | 0.883252 | 0.134474 | 15 | `tuned_threshold` |
| 0.05 | 3-block, H=36 | `knn_gain` | `latent_only` | 0.866504 | 0.371638 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=36 | `knn_gain` | `margin` | 0.858924 | 0.229829 | 25 | `gain_threshold` |
| 0.05 | 3-block, H=36 | `knn_gain` | `uncertainty` | 0.832885 | 0.310513 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=36 | `rescue_harm` | `context` | 0.861247 | 0.232274 | 100 | `thr=0.0300,cap=0.2000` |
| 0.05 | 3-block, H=36 | `rescue_harm` | `context_latent` | 0.871638 | 0.171149 | 25 | `thr=0.0500,cap=0.1600` |
| 0.05 | 3-block, H=36 | `rescue_harm` | `latent_only` | 0.866504 | 0.322738 | 100 | `thr=0.0500,cap=0.0500` |
| 0.05 | 3-block, H=36 | `rescue_harm` | `margin` | 0.864792 | 0.210269 | 25 | `thr=0.0800,cap=0.2800` |
| 0.05 | 3-block, H=36 | `rescue_harm` | `uncertainty` | 0.864670 | 0.163814 | 50 | `thr=0.1000,cap=0.2000` |
| 0.10 | 3-block, H=36 | `knn_gain` | `context` | 0.867971 | 0.146699 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=36 | `knn_gain` | `context_latent` | 0.876528 | 0.134474 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=36 | `knn_gain` | `latent_only` | 0.870171 | 0.222494 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=36 | `knn_gain` | `margin` | 0.847433 | 0.229829 | 25 | `tuned_threshold` |
| 0.10 | 3-block, H=36 | `knn_gain` | `uncertainty` | 0.844254 | 0.212714 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=36 | `rescue_harm` | `context` | 0.884108 | 0.107579 | 15 | `thr=0.1333,cap=0.1333` |
| 0.10 | 3-block, H=36 | `rescue_harm` | `context_latent` | 0.884108 | 0.107579 | 15 | `thr=0.1333,cap=0.1333` |
| 0.10 | 3-block, H=36 | `rescue_harm` | `latent_only` | 0.884108 | 0.107579 | 15 | `thr=0.1333,cap=0.0000` |
| 0.10 | 3-block, H=36 | `rescue_harm` | `margin` | 0.867482 | 0.127139 | 200 | `thr=0.0900,cap=0.1050` |
| 0.10 | 3-block, H=36 | `rescue_harm` | `uncertainty` | 0.877506 | 0.124694 | 50 | `thr=0.1800,cap=0.1200` |
| 0.20 | 3-block, H=36 | `knn_gain` | `context` | 0.880196 | 0.085575 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=36 | `knn_gain` | `context_latent` | 0.881663 | 0.078240 | 15 | `gain_threshold` |
| 0.20 | 3-block, H=36 | `knn_gain` | `latent_only` | 0.880196 | 0.061125 | 15 | `gain_threshold` |
| 0.20 | 3-block, H=36 | `knn_gain` | `margin` | 0.866504 | 0.105134 | 15 | `gain_threshold` |
| 0.20 | 3-block, H=36 | `knn_gain` | `uncertainty` | 0.856724 | 0.141809 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=36 | `rescue_harm` | `context` | 0.877751 | 0.036675 | 15 | `thr=0.2667,cap=0.1333` |
| 0.20 | 3-block, H=36 | `rescue_harm` | `context_latent` | 0.878729 | 0.092910 | 15 | `thr=0.2000,cap=0.1333` |
| 0.20 | 3-block, H=36 | `rescue_harm` | `latent_only` | 0.888020 | 0.034230 | 15 | `thr=0.3333,cap=0.0000` |
| 0.20 | 3-block, H=36 | `rescue_harm` | `margin` | 0.858191 | 0.122249 | 200 | `thr=0.1300,cap=0.1050` |
| 0.20 | 3-block, H=36 | `rescue_harm` | `uncertainty` | 0.865037 | 0.124694 | 50 | `thr=0.1800,cap=0.1200` |
| 0.30 | 3-block, H=36 | `knn_gain` | `context` | 0.887286 | 0.017115 | 15 | `gain_threshold` |
| 0.30 | 3-block, H=36 | `knn_gain` | `context_latent` | 0.873839 | 0.078240 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=36 | `knn_gain` | `latent_only` | 0.874083 | 0.061125 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=36 | `knn_gain` | `margin` | 0.876528 | 0.036675 | 50 | `gain_threshold` |
| 0.30 | 3-block, H=36 | `knn_gain` | `uncertainty` | 0.886553 | 0.019560 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=36 | `rescue_harm` | `context` | 0.883374 | 0.022005 | 100 | `thr=0.1442,cap=0.0700` |
| 0.30 | 3-block, H=36 | `rescue_harm` | `context_latent` | 0.882641 | 0.048900 | 25 | `thr=0.2800,cap=0.1600` |
| 0.30 | 3-block, H=36 | `rescue_harm` | `latent_only` | 0.884597 | 0.034230 | 15 | `thr=0.3000,cap=0.0000` |
| 0.30 | 3-block, H=36 | `rescue_harm` | `margin` | 0.869193 | 0.061125 | 200 | `thr=0.2000,cap=0.1050` |
| 0.30 | 3-block, H=36 | `rescue_harm` | `uncertainty` | 0.897311 | 0.000000 | 25 | `thr=0.4000,cap=0.0400` |
| 0.50 | 3-block, H=36 | `knn_gain` | `context` | 0.897311 | 0.000000 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=36 | `knn_gain` | `context_latent` | 0.883863 | 0.026895 | 25 | `tuned_threshold` |
| 0.50 | 3-block, H=36 | `knn_gain` | `latent_only` | 0.896088 | 0.002445 | 15 | `tuned_threshold` |
| 0.50 | 3-block, H=36 | `knn_gain` | `margin` | 0.897311 | 0.000000 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=36 | `knn_gain` | `uncertainty` | 0.882641 | 0.019560 | 15 | `tuned_threshold` |
| 0.50 | 3-block, H=36 | `rescue_harm` | `context` | 0.897311 | 0.000000 | 25 | `thr=0.5000,cap=0.0000` |
| 0.50 | 3-block, H=36 | `rescue_harm` | `context_latent` | 0.883863 | 0.026895 | 25 | `thr=0.3600,cap=0.1600` |
| 0.50 | 3-block, H=36 | `rescue_harm` | `latent_only` | 0.897311 | 0.000000 | 15 | `thr=0.5000,cap=0.0000` |
| 0.50 | 3-block, H=36 | `rescue_harm` | `margin` | 0.880196 | 0.014670 | 100 | `thr=0.2900,cap=0.0500` |
| 0.50 | 3-block, H=36 | `rescue_harm` | `uncertainty` | 0.897311 | 0.000000 | 15 | `thr=0.5000,cap=0.0000` |
| 0.02 | 3-block, H=48 | `knn_gain` | `context` | 0.866219 | 0.097015 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=48 | `knn_gain` | `context_latent` | 0.865920 | 0.236318 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=48 | `knn_gain` | `latent_only` | 0.862388 | 0.412935 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=48 | `knn_gain` | `margin` | 0.846716 | 0.201493 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=48 | `knn_gain` | `uncertainty` | 0.842189 | 0.303483 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=48 | `rescue_harm` | `context` | 0.870945 | 0.109453 | 25 | `thr=0.0400,cap=0.2000` |
| 0.02 | 3-block, H=48 | `rescue_harm` | `context_latent` | 0.868607 | 0.226368 | 15 | `thr=0.0200,cap=0.2000` |
| 0.02 | 3-block, H=48 | `rescue_harm` | `latent_only` | 0.846269 | 0.348259 | 50 | `thr=0.0400,cap=0.0600` |
| 0.02 | 3-block, H=48 | `rescue_harm` | `margin` | 0.853035 | 0.258706 | 25 | `thr=0.0200,cap=0.2000` |
| 0.02 | 3-block, H=48 | `rescue_harm` | `uncertainty` | 0.859552 | 0.181592 | 200 | `thr=0.0250,cap=0.1600` |
| 0.05 | 3-block, H=48 | `knn_gain` | `context` | 0.863308 | 0.097015 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=48 | `knn_gain` | `context_latent` | 0.866791 | 0.126866 | 15 | `tuned_threshold` |
| 0.05 | 3-block, H=48 | `knn_gain` | `latent_only` | 0.862811 | 0.256219 | 15 | `tuned_threshold` |
| 0.05 | 3-block, H=48 | `knn_gain` | `margin` | 0.840672 | 0.201493 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=48 | `knn_gain` | `uncertainty` | 0.833085 | 0.303483 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=48 | `rescue_harm` | `context` | 0.866418 | 0.084577 | 25 | `thr=0.0400,cap=0.2000` |
| 0.05 | 3-block, H=48 | `rescue_harm` | `context_latent` | 0.863557 | 0.141791 | 25 | `thr=0.0800,cap=0.3500` |
| 0.05 | 3-block, H=48 | `rescue_harm` | `latent_only` | 0.854478 | 0.223881 | 25 | `thr=0.1200,cap=0.0800` |
| 0.05 | 3-block, H=48 | `rescue_harm` | `margin` | 0.862935 | 0.104478 | 50 | `thr=0.1200,cap=0.2000` |
| 0.05 | 3-block, H=48 | `rescue_harm` | `uncertainty` | 0.866045 | 0.092040 | 200 | `thr=0.0500,cap=0.1600` |
| 0.10 | 3-block, H=48 | `knn_gain` | `context` | 0.851244 | 0.044776 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=48 | `knn_gain` | `context_latent` | 0.860448 | 0.126866 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=48 | `knn_gain` | `latent_only` | 0.850000 | 0.256219 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=48 | `knn_gain` | `margin` | 0.830597 | 0.201493 | 15 | `tuned_threshold` |
| 0.10 | 3-block, H=48 | `knn_gain` | `uncertainty` | 0.842040 | 0.111940 | 25 | `gain_threshold` |
| 0.10 | 3-block, H=48 | `rescue_harm` | `context` | 0.850746 | 0.024876 | 15 | `thr=0.1333,cap=0.1333` |
| 0.10 | 3-block, H=48 | `rescue_harm` | `context_latent` | 0.855721 | 0.099502 | 25 | `thr=0.1000,cap=0.3500` |
| 0.10 | 3-block, H=48 | `rescue_harm` | `latent_only` | 0.856965 | 0.136816 | 15 | `thr=0.2000,cap=0.1333` |
| 0.10 | 3-block, H=48 | `rescue_harm` | `margin` | 0.857711 | 0.104478 | 50 | `thr=0.1200,cap=0.2000` |
| 0.10 | 3-block, H=48 | `rescue_harm` | `uncertainty` | 0.861443 | 0.092040 | 200 | `thr=0.0450,cap=0.1600` |
| 0.20 | 3-block, H=48 | `knn_gain` | `context` | 0.854229 | 0.007463 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=48 | `knn_gain` | `context_latent` | 0.847761 | 0.126866 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=48 | `knn_gain` | `latent_only` | 0.845771 | 0.062189 | 15 | `gain_threshold` |
| 0.20 | 3-block, H=48 | `knn_gain` | `margin` | 0.843284 | 0.049751 | 15 | `gain_threshold` |
| 0.20 | 3-block, H=48 | `knn_gain` | `uncertainty` | 0.848259 | 0.037313 | 50 | `gain_threshold` |
| 0.20 | 3-block, H=48 | `rescue_harm` | `context` | 0.853731 | 0.009950 | 15 | `thr=0.2667,cap=0.1333` |
| 0.20 | 3-block, H=48 | `rescue_harm` | `context_latent` | 0.847761 | 0.052239 | 25 | `thr=0.1600,cap=0.1600` |
| 0.20 | 3-block, H=48 | `rescue_harm` | `latent_only` | 0.853234 | 0.000000 | 15 | `thr=0.4667,cap=0.0000` |
| 0.20 | 3-block, H=48 | `rescue_harm` | `margin` | 0.844776 | 0.017413 | 25 | `thr=0.2800,cap=0.0400` |
| 0.20 | 3-block, H=48 | `rescue_harm` | `uncertainty` | 0.852239 | 0.017413 | 25 | `thr=0.2000,cap=0.0000` |
| 0.30 | 3-block, H=48 | `knn_gain` | `context` | 0.853483 | 0.007463 | 15 | `gain_threshold` |
| 0.30 | 3-block, H=48 | `knn_gain` | `context_latent` | 0.851741 | 0.029851 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=48 | `knn_gain` | `latent_only` | 0.839552 | 0.062189 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=48 | `knn_gain` | `margin` | 0.837562 | 0.027363 | 15 | `gain_threshold` |
| 0.30 | 3-block, H=48 | `knn_gain` | `uncertainty` | 0.848507 | 0.007463 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=48 | `rescue_harm` | `context` | 0.851741 | 0.004975 | 25 | `thr=0.2000,cap=0.0800` |
| 0.30 | 3-block, H=48 | `rescue_harm` | `context_latent` | 0.853234 | 0.024876 | 50 | `thr=0.1600,cap=0.1600` |
| 0.30 | 3-block, H=48 | `rescue_harm` | `latent_only` | 0.853234 | 0.000000 | 15 | `thr=0.4667,cap=0.0000` |
| 0.30 | 3-block, H=48 | `rescue_harm` | `margin` | 0.853234 | 0.000000 | 200 | `thr=0.0750,cap=0.0900` |
| 0.30 | 3-block, H=48 | `rescue_harm` | `uncertainty` | 0.850498 | 0.017413 | 25 | `thr=0.2000,cap=0.0000` |
| 0.50 | 3-block, H=48 | `knn_gain` | `context` | 0.853234 | 0.000000 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=48 | `knn_gain` | `context_latent` | 0.849502 | 0.022388 | 15 | `tuned_threshold` |
| 0.50 | 3-block, H=48 | `knn_gain` | `latent_only` | 0.853234 | 0.000000 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=48 | `knn_gain` | `margin` | 0.853234 | 0.000000 | 15 | `tuned_threshold` |
| 0.50 | 3-block, H=48 | `knn_gain` | `uncertainty` | 0.847015 | 0.007463 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=48 | `rescue_harm` | `context` | 0.853234 | 0.000000 | 50 | `thr=0.3400,cap=0.0800` |
| 0.50 | 3-block, H=48 | `rescue_harm` | `context_latent` | 0.854478 | 0.007463 | 25 | `thr=0.3467,cap=0.1600` |
| 0.50 | 3-block, H=48 | `rescue_harm` | `latent_only` | 0.853234 | 0.000000 | 15 | `thr=0.4667,cap=0.0000` |
| 0.50 | 3-block, H=48 | `rescue_harm` | `margin` | 0.853234 | 0.000000 | 200 | `thr=0.0750,cap=0.0900` |
| 0.50 | 3-block, H=48 | `rescue_harm` | `uncertainty` | 0.847015 | 0.017413 | 25 | `thr=0.2000,cap=0.0000` |
| 0.02 | 3-block, H=72 | `knn_gain` | `context` | 0.806538 | 0.472155 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=72 | `knn_gain` | `context_latent` | 0.805763 | 0.268765 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=72 | `knn_gain` | `latent_only` | 0.774867 | 0.239709 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=72 | `knn_gain` | `margin` | 0.752494 | 0.268765 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=72 | `knn_gain` | `uncertainty` | 0.770266 | 0.227603 | 15 | `gain_threshold` |
| 0.02 | 3-block, H=72 | `rescue_harm` | `context` | 0.801840 | 0.464891 | 15 | `thr=0.0200,cap=0.1333` |
| 0.02 | 3-block, H=72 | `rescue_harm` | `context_latent` | 0.796368 | 0.254237 | 15 | `thr=0.0667,cap=0.2000` |
| 0.02 | 3-block, H=72 | `rescue_harm` | `latent_only` | 0.744455 | 0.065375 | 100 | `thr=0.0300,cap=0.0200` |
| 0.02 | 3-block, H=72 | `rescue_harm` | `margin` | 0.769201 | 0.159806 | 100 | `thr=0.0700,cap=0.2000` |
| 0.02 | 3-block, H=72 | `rescue_harm` | `uncertainty` | 0.775690 | 0.198547 | 100 | `thr=0.0200,cap=0.2000` |
| 0.05 | 3-block, H=72 | `knn_gain` | `context` | 0.779661 | 0.338983 | 15 | `tuned_threshold` |
| 0.05 | 3-block, H=72 | `knn_gain` | `context_latent` | 0.797700 | 0.268765 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=72 | `knn_gain` | `latent_only` | 0.767676 | 0.239709 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=72 | `knn_gain` | `margin` | 0.744431 | 0.268765 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=72 | `knn_gain` | `uncertainty` | 0.763438 | 0.227603 | 15 | `gain_threshold` |
| 0.05 | 3-block, H=72 | `rescue_harm` | `context` | 0.787893 | 0.464891 | 15 | `thr=0.0500,cap=0.1333` |
| 0.05 | 3-block, H=72 | `rescue_harm` | `context_latent` | 0.793826 | 0.249395 | 50 | `thr=0.0400,cap=0.3500` |
| 0.05 | 3-block, H=72 | `rescue_harm` | `latent_only` | 0.733414 | 0.004843 | 50 | `thr=0.1000,cap=0.0000` |
| 0.05 | 3-block, H=72 | `rescue_harm` | `margin` | 0.729661 | 0.079903 | 50 | `thr=0.0400,cap=0.1200` |
| 0.05 | 3-block, H=72 | `rescue_harm` | `uncertainty` | 0.769734 | 0.198547 | 100 | `thr=0.0200,cap=0.2000` |
| 0.10 | 3-block, H=72 | `knn_gain` | `context` | 0.762712 | 0.338983 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=72 | `knn_gain` | `context_latent` | 0.780387 | 0.162228 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=72 | `knn_gain` | `latent_only` | 0.741404 | 0.043584 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=72 | `knn_gain` | `margin` | 0.748910 | 0.162228 | 15 | `gain_threshold` |
| 0.10 | 3-block, H=72 | `knn_gain` | `uncertainty` | 0.742131 | 0.084746 | 50 | `tuned_threshold` |
| 0.10 | 3-block, H=72 | `rescue_harm` | `context` | 0.760291 | 0.314770 | 100 | `thr=0.0500,cap=0.1800` |
| 0.10 | 3-block, H=72 | `rescue_harm` | `context_latent` | 0.781356 | 0.249395 | 50 | `thr=0.0400,cap=0.3500` |
| 0.10 | 3-block, H=72 | `rescue_harm` | `latent_only` | 0.733172 | 0.004843 | 50 | `thr=0.1000,cap=0.0000` |
| 0.10 | 3-block, H=72 | `rescue_harm` | `margin` | 0.739952 | 0.033898 | 50 | `thr=0.1400,cap=0.1200` |
| 0.10 | 3-block, H=72 | `rescue_harm` | `uncertainty` | 0.747700 | 0.077482 | 50 | `thr=0.1000,cap=0.1500` |
| 0.20 | 3-block, H=72 | `knn_gain` | `context` | 0.728814 | 0.338983 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=72 | `knn_gain` | `context_latent` | 0.745763 | 0.084746 | 15 | `tuned_threshold` |
| 0.20 | 3-block, H=72 | `knn_gain` | `latent_only` | 0.731235 | 0.000000 | 15 | `gain_threshold` |
| 0.20 | 3-block, H=72 | `knn_gain` | `margin` | 0.743341 | 0.060533 | 50 | `tuned_threshold` |
| 0.20 | 3-block, H=72 | `knn_gain` | `uncertainty` | 0.736077 | 0.024213 | 50 | `tuned_threshold` |
| 0.20 | 3-block, H=72 | `rescue_harm` | `context` | 0.727845 | 0.162228 | 100 | `thr=0.0900,cap=0.1800` |
| 0.20 | 3-block, H=72 | `rescue_harm` | `context_latent` | 0.734140 | 0.058111 | 200 | `thr=0.0524,cap=0.1200` |
| 0.20 | 3-block, H=72 | `rescue_harm` | `latent_only` | 0.731235 | 0.000000 | 15 | `thr=1000000000.0000,cap=0.0000` |
| 0.20 | 3-block, H=72 | `rescue_harm` | `margin` | 0.726392 | 0.012107 | 50 | `thr=0.1800,cap=0.1200` |
| 0.20 | 3-block, H=72 | `rescue_harm` | `uncertainty` | 0.736077 | 0.012107 | 50 | `thr=0.2000,cap=0.0800` |
| 0.30 | 3-block, H=72 | `knn_gain` | `context` | 0.724939 | 0.150121 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=72 | `knn_gain` | `context_latent` | 0.737288 | 0.084746 | 15 | `tuned_threshold` |
| 0.30 | 3-block, H=72 | `knn_gain` | `latent_only` | 0.731235 | 0.000000 | 50 | `tuned_threshold` |
| 0.30 | 3-block, H=72 | `knn_gain` | `margin` | 0.725182 | 0.012107 | 25 | `gain_threshold` |
| 0.30 | 3-block, H=72 | `knn_gain` | `uncertainty` | 0.730508 | 0.002421 | 50 | `tuned_threshold` |
| 0.30 | 3-block, H=72 | `rescue_harm` | `context` | 0.712591 | 0.142857 | 100 | `thr=0.0900,cap=0.1500` |
| 0.30 | 3-block, H=72 | `rescue_harm` | `context_latent` | 0.732930 | 0.002421 | 50 | `thr=0.2600,cap=0.0800` |
| 0.30 | 3-block, H=72 | `rescue_harm` | `latent_only` | 0.731235 | 0.000000 | 15 | `thr=1000000000.0000,cap=0.0000` |
| 0.30 | 3-block, H=72 | `rescue_harm` | `margin` | 0.727361 | 0.004843 | 25 | `thr=0.3000,cap=0.0000` |
| 0.30 | 3-block, H=72 | `rescue_harm` | `uncertainty` | 0.731235 | 0.000000 | 15 | `thr=0.3333,cap=0.0000` |
| 0.50 | 3-block, H=72 | `knn_gain` | `context` | 0.732446 | 0.002421 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=72 | `knn_gain` | `context_latent` | 0.721550 | 0.048426 | 15 | `tuned_threshold` |
| 0.50 | 3-block, H=72 | `knn_gain` | `latent_only` | 0.731235 | 0.000000 | 15 | `gain_threshold` |
| 0.50 | 3-block, H=72 | `knn_gain` | `margin` | 0.728814 | 0.004843 | 25 | `tuned_threshold` |
| 0.50 | 3-block, H=72 | `knn_gain` | `uncertainty` | 0.730024 | 0.002421 | 50 | `tuned_threshold` |
| 0.50 | 3-block, H=72 | `rescue_harm` | `context` | 0.732446 | 0.002421 | 15 | `thr=0.5000,cap=0.0667` |
| 0.50 | 3-block, H=72 | `rescue_harm` | `context_latent` | 0.731235 | 0.000000 | 50 | `thr=0.2600,cap=0.0400` |
| 0.50 | 3-block, H=72 | `rescue_harm` | `latent_only` | 0.731235 | 0.000000 | 15 | `thr=0.5000,cap=0.0000` |
| 0.50 | 3-block, H=72 | `rescue_harm` | `margin` | 0.731235 | 0.000000 | 15 | `thr=0.5000,cap=0.0000` |
| 0.50 | 3-block, H=72 | `rescue_harm` | `uncertainty` | 0.731235 | 0.000000 | 15 | `thr=0.3333,cap=0.0000` |

## Interpretation guide

- If `latent_only` helps, current visual-state complexity contains value-of-computation signal.
- If `context_latent` beats `context`, latent state complexity improves routing beyond uncertainty and horizon/velocity.
- If it does not help, the current cheap-model risk features are already the dominant observable signal.

# SPSM v8 geometry feature audit

This audit checks whether the added geometry/OOD features actually contain signal for the value-of-computation target.
The target is `route = 1[full_correct - small_correct > lambda]`.

## Dataset

- rows after small/full merge: `2853`
- context features: `11`
- added geometry features: `28`

Added geometry features:
- `blocked_norm`
- `blocked_x_horizon`
- `blocked_x_velocity`
- `complexity_score`
- `uncertainty_x_blocked`
- `uncertainty_x_horizon`
- `uncertainty_x_velocity`
- `margin_x_blocked`
- `best_dist_x_blocked`
- `second_dist_x_blocked`
- `npz_anchor_blocked`
- `npz_candidate_blocked_mean`
- `npz_candidate_blocked_std`
- `npz_blocked_mismatch_frac`
- `npz_state_dist_min`
- `npz_state_dist_mean`
- `npz_state_dist_max`
- `npz_state_dist_std`
- `npz_same_state_neg_frac`
- `npz_same_action_neg_frac`
- `action_1_x_blocked`
- `action_1_x_horizon`
- `action_2_x_blocked`
- `action_2_x_horizon`
- `action_3_x_blocked`
- `action_3_x_horizon`
- `action_4_x_blocked`
- `action_4_x_horizon`

## Feature standard deviations by variant


### ID 2-block, H=36

| feature | std |
|---|---:|
| `blocked_norm` | 0 |
| `blocked_x_horizon` | 0 |
| `blocked_x_velocity` | 0 |
| `complexity_score` | 0 |
| `uncertainty_x_blocked` | 0 |
| `uncertainty_x_horizon` | 0 |
| `uncertainty_x_velocity` | 0 |
| `margin_x_blocked` | 0 |
| `best_dist_x_blocked` | 0 |
| `second_dist_x_blocked` | 0 |
| `npz_state_dist_min` | 0 |
| `action_2_x_blocked` | 0 |
| `action_1_x_blocked` | 0 |
| `action_1_x_horizon` | 0 |
| `npz_same_state_neg_frac` | 0 |
| `npz_same_action_neg_frac` | 0 |
| `action_4_x_blocked` | 0 |
| `action_4_x_horizon` | 0 |
| `action_3_x_horizon` | 0 |
| `action_3_x_blocked` | 0 |
| `action_2_x_horizon` | 0 |
| `npz_state_dist_mean` | 0.0101999 |
| `npz_state_dist_std` | 0.0115031 |
| `npz_state_dist_max` | 0.0252462 |
| `npz_candidate_blocked_std` | 0.0412836 |
| `npz_blocked_mismatch_frac` | 0.116148 |
| `npz_candidate_blocked_mean` | 0.181585 |
| `npz_anchor_blocked` | 0.499481 |

### 2-block, H=72

| feature | std |
|---|---:|
| `blocked_norm` | 0 |
| `blocked_x_horizon` | 0 |
| `blocked_x_velocity` | 0 |
| `complexity_score` | 0 |
| `uncertainty_x_blocked` | 0 |
| `uncertainty_x_velocity` | 0 |
| `margin_x_blocked` | 0 |
| `best_dist_x_blocked` | 0 |
| `npz_state_dist_min` | 0 |
| `second_dist_x_blocked` | 0 |
| `action_3_x_blocked` | 0 |
| `action_4_x_blocked` | 0 |
| `action_1_x_blocked` | 0 |
| `action_2_x_blocked` | 0 |
| `npz_same_state_neg_frac` | 0 |
| `npz_same_action_neg_frac` | 0 |
| `npz_state_dist_mean` | 0.0107082 |
| `npz_state_dist_std` | 0.0122277 |
| `npz_state_dist_max` | 0.0266939 |
| `npz_candidate_blocked_std` | 0.0422209 |
| `npz_blocked_mismatch_frac` | 0.117606 |
| `uncertainty_x_horizon` | 0.123602 |
| `npz_candidate_blocked_mean` | 0.182578 |
| `action_1_x_horizon` | 0.423361 |
| `action_2_x_horizon` | 0.427832 |
| `action_4_x_horizon` | 0.434915 |
| `action_3_x_horizon` | 0.446566 |
| `npz_anchor_blocked` | 0.500596 |

### 2-block, V=1.8

| feature | std |
|---|---:|
| `blocked_norm` | 0 |
| `blocked_x_horizon` | 0 |
| `blocked_x_velocity` | 0 |
| `complexity_score` | 0 |
| `uncertainty_x_blocked` | 0 |
| `uncertainty_x_horizon` | 0 |
| `margin_x_blocked` | 0 |
| `best_dist_x_blocked` | 0 |
| `npz_state_dist_min` | 0 |
| `second_dist_x_blocked` | 0 |
| `action_4_x_blocked` | 0 |
| `action_4_x_horizon` | 0 |
| `action_1_x_blocked` | 0 |
| `action_1_x_horizon` | 0 |
| `action_2_x_blocked` | 0 |
| `action_2_x_horizon` | 0 |
| `action_3_x_horizon` | 0 |
| `action_3_x_blocked` | 0 |
| `npz_same_action_neg_frac` | 5.55817e-17 |
| `npz_same_state_neg_frac` | 5.55817e-17 |
| `npz_state_dist_mean` | 0.0103722 |
| `npz_state_dist_std` | 0.0119353 |
| `npz_state_dist_max` | 0.0266037 |
| `npz_candidate_blocked_std` | 0.0403795 |
| `uncertainty_x_velocity` | 0.0809378 |
| `npz_blocked_mismatch_frac` | 0.115348 |
| `npz_candidate_blocked_mean` | 0.177178 |
| `npz_anchor_blocked` | 0.49937 |

### 3-block, H=36

| feature | std |
|---|---:|
| `blocked_norm` | 0 |
| `blocked_x_horizon` | 0 |
| `blocked_x_velocity` | 0 |
| `complexity_score` | 0 |
| `uncertainty_x_horizon` | 0 |
| `uncertainty_x_velocity` | 0 |
| `npz_state_dist_min` | 0 |
| `action_2_x_horizon` | 0 |
| `action_3_x_horizon` | 0 |
| `action_4_x_horizon` | 0 |
| `action_1_x_horizon` | 0 |
| `npz_same_action_neg_frac` | 5.55791e-17 |
| `npz_same_state_neg_frac` | 5.55791e-17 |
| `npz_state_dist_mean` | 0.0108515 |
| `npz_state_dist_std` | 0.0125803 |
| `npz_state_dist_max` | 0.02738 |
| `npz_candidate_blocked_std` | 0.0652441 |
| `margin_x_blocked` | 0.0660595 |
| `npz_blocked_mismatch_frac` | 0.160877 |
| `uncertainty_x_blocked` | 0.165059 |
| `npz_candidate_blocked_mean` | 0.189972 |
| `second_dist_x_blocked` | 0.214276 |
| `best_dist_x_blocked` | 0.220785 |
| `action_4_x_blocked` | 0.419656 |
| `npz_anchor_blocked` | 0.425864 |
| `action_3_x_blocked` | 0.425864 |
| `action_1_x_blocked` | 0.428851 |
| `action_2_x_blocked` | 0.455877 |

### 3-block, H=48

| feature | std |
|---|---:|
| `blocked_norm` | 0 |
| `blocked_x_horizon` | 0 |
| `blocked_x_velocity` | 0 |
| `complexity_score` | 0 |
| `uncertainty_x_velocity` | 0 |
| `npz_state_dist_min` | 0 |
| `npz_same_state_neg_frac` | 0 |
| `npz_same_action_neg_frac` | 0 |
| `npz_state_dist_mean` | 0.0102138 |
| `npz_state_dist_std` | 0.0118673 |
| `npz_state_dist_max` | 0.0265821 |
| `uncertainty_x_horizon` | 0.0482821 |
| `npz_candidate_blocked_std` | 0.0607375 |
| `margin_x_blocked` | 0.0674877 |
| `action_1_x_horizon` | 0.131982 |
| `uncertainty_x_blocked` | 0.144846 |
| `action_2_x_horizon` | 0.145229 |
| `action_4_x_horizon` | 0.146611 |
| `action_3_x_horizon` | 0.151603 |
| `npz_blocked_mismatch_frac` | 0.160084 |
| `npz_candidate_blocked_mean` | 0.180994 |
| `second_dist_x_blocked` | 0.216156 |
| `best_dist_x_blocked` | 0.222485 |
| `npz_anchor_blocked` | 0.394005 |
| `action_1_x_blocked` | 0.395945 |
| `action_2_x_blocked` | 0.435688 |
| `action_4_x_blocked` | 0.439833 |
| `action_3_x_blocked` | 0.45481 |

### 3-block, H=72

| feature | std |
|---|---:|
| `blocked_norm` | 0 |
| `blocked_x_horizon` | 0 |
| `blocked_x_velocity` | 0 |
| `complexity_score` | 0 |
| `uncertainty_x_velocity` | 0 |
| `npz_state_dist_min` | 0 |
| `npz_same_state_neg_frac` | 5.55785e-17 |
| `npz_same_action_neg_frac` | 5.55785e-17 |
| `npz_state_dist_mean` | 0.0103621 |
| `npz_state_dist_std` | 0.0117323 |
| `npz_state_dist_max` | 0.0259317 |
| `npz_candidate_blocked_std` | 0.0606365 |
| `margin_x_blocked` | 0.0782315 |
| `npz_blocked_mismatch_frac` | 0.163642 |
| `uncertainty_x_blocked` | 0.183857 |
| `uncertainty_x_horizon` | 0.183857 |
| `npz_candidate_blocked_mean` | 0.183863 |
| `second_dist_x_blocked` | 0.238462 |
| `best_dist_x_blocked` | 0.247719 |
| `action_4_x_blocked` | 0.425936 |
| `action_4_x_horizon` | 0.425936 |
| `action_2_x_horizon` | 0.431774 |
| `action_2_x_blocked` | 0.431774 |
| `action_3_x_horizon` | 0.437319 |
| `action_3_x_blocked` | 0.437319 |
| `action_1_x_blocked` | 0.438661 |
| `action_1_x_horizon` | 0.438661 |
| `npz_anchor_blocked` | 0.445111 |

## Top feature signal at lambda = 0.10


### ALL

| feature | std | corr(gain) | corr(route) | AUC(route) | mean route=1 | mean route=0 |
|---|---:|---:|---:|---:|---:|---:|
| `cheap_confidence` | 0.1356 | -0.1232 | -0.3969 | 0.0978 | 0.6908 | 0.9425 |
| `cheap_uncertainty` | 0.1356 | 0.1232 | 0.3969 | 0.9022 | 0.3092 | 0.05746 |
| `cheap_pred_margin` | 0.0747 | -0.0585 | -0.2819 | 0.0979 | 0.02839 | 0.1269 |
| `uncertainty_x_horizon` | 0.1017 | 0.2034 | 0.3696 | 0.8291 | 0.2032 | 0.0275 |
| `uncertainty_x_blocked` | 0.1255 | 0.0681 | 0.3495 | 0.8078 | 0.2486 | 0.04347 |
| `complexity_score` | 0.7291 | 0.0351 | 0.2132 | 0.7495 | 1.382 | 0.6549 |
| `horizon_norm` | 0.4385 | 0.1215 | 0.1910 | 0.7307 | 0.7105 | 0.3187 |
| `blocked_x_horizon` | 0.3516 | 0.0997 | 0.2334 | 0.7294 | 0.5572 | 0.1733 |
| `blocked_norm` | 0.4949 | -0.0185 | 0.1730 | 0.7002 | 0.8102 | 0.4098 |
| `npz_anchor_blocked` | 0.4896 | 0.0096 | 0.1627 | 0.6863 | 0.9562 | 0.5836 |
| `best_dist_x_blocked` | 0.9376 | -0.0308 | 0.1647 | 0.6826 | 1.49 | 0.7673 |
| `second_dist_x_blocked` | 0.9806 | -0.0321 | 0.1540 | 0.6557 | 1.513 | 0.8065 |
| `cheap_second_best_distance` | 0.2541 | -0.0631 | -0.0909 | 0.3704 | 1.882 | 1.99 |
| `action_1_x_horizon` | 0.2608 | 0.1152 | 0.1715 | 0.6215 | 0.2798 | 0.07057 |
| `npz_candidate_blocked_std` | 0.05133 | 0.0059 | 0.0190 | 0.6196 | 0.466 | 0.4614 |

### 3-block, H=36

| feature | std | corr(gain) | corr(route) | AUC(route) | mean route=1 | mean route=0 |
|---|---:|---:|---:|---:|---:|---:|
| `cheap_pred_margin` | 0.06598 | 0.1199 | -0.2450 | 0.0858 | 0.014 | 0.09281 |
| `margin_x_blocked` | 0.06598 | 0.1199 | -0.2450 | 0.0858 | 0.014 | 0.09281 |
| `cheap_confidence` | 0.1649 | 0.1023 | -0.3595 | 0.0943 | 0.6057 | 0.8947 |
| `cheap_uncertainty` | 0.1649 | -0.1023 | 0.3595 | 0.9057 | 0.3943 | 0.1053 |
| `uncertainty_x_blocked` | 0.1649 | -0.1023 | 0.3595 | 0.9057 | 0.3943 | 0.1053 |
| `npz_candidate_blocked_mean` | 0.1897 | 0.0363 | -0.1592 | 0.2775 | 0.3556 | 0.5028 |
| `action_2` | 0.4553 | 0.0167 | 0.1235 | 0.6371 | 0.5556 | 0.2813 |
| `action_2_x_blocked` | 0.4553 | 0.0167 | 0.1235 | 0.6371 | 0.5556 | 0.2813 |
| `npz_anchor_blocked` | 0.4253 | -0.0980 | 0.1196 | 0.6240 | 1 | 0.7519 |
| `action_4` | 0.4191 | 0.0804 | -0.0880 | 0.4101 | 0.05556 | 0.2353 |
| `action_4_x_blocked` | 0.4191 | 0.0804 | -0.0880 | 0.4101 | 0.05556 | 0.2353 |
| `cheap_second_best_distance` | 0.214 | 0.0014 | -0.0638 | 0.4169 | 1.91 | 1.976 |
| `second_dist_x_blocked` | 0.214 | 0.0014 | -0.0638 | 0.4169 | 1.91 | 1.976 |
| `npz_blocked_mismatch_frac` | 0.1607 | 0.0636 | 0.0580 | 0.5815 | 0.6444 | 0.599 |
| `action_3` | 0.4253 | -0.0053 | -0.0636 | 0.4341 | 0.1111 | 0.243 |

### 3-block, H=48

| feature | std | corr(gain) | corr(route) | AUC(route) | mean route=1 | mean route=0 |
|---|---:|---:|---:|---:|---:|---:|
| `cheap_pred_margin` | 0.0674 | -0.0675 | -0.2655 | 0.1100 | 0.02433 | 0.09842 |
| `margin_x_blocked` | 0.0674 | -0.0675 | -0.2655 | 0.1100 | 0.02433 | 0.09842 |
| `uncertainty_x_horizon` | 0.04822 | 0.1641 | 0.3908 | 0.8798 | 0.1059 | 0.02789 |
| `cheap_confidence` | 0.1447 | -0.1641 | -0.3908 | 0.1202 | 0.6822 | 0.9163 |
| `cheap_uncertainty` | 0.1447 | 0.1641 | 0.3908 | 0.8798 | 0.3178 | 0.08366 |
| `uncertainty_x_blocked` | 0.1447 | 0.1641 | 0.3908 | 0.8798 | 0.3178 | 0.08366 |
| `npz_blocked_mismatch_frac` | 0.1599 | -0.0788 | -0.1314 | 0.3848 | 0.504 | 0.591 |
| `action_1_x_horizon` | 0.1318 | 0.0846 | 0.1341 | 0.6098 | 0.1333 | 0.06012 |
| `action_1` | 0.3955 | 0.0846 | 0.1341 | 0.6098 | 0.4 | 0.1804 |
| `action_1_x_blocked` | 0.3955 | 0.0846 | 0.1341 | 0.6098 | 0.4 | 0.1804 |
| `npz_anchor_blocked` | 0.3935 | -0.0164 | 0.1253 | 0.6021 | 1 | 0.7958 |
| `cheap_second_best_distance` | 0.2159 | -0.0716 | -0.0560 | 0.4066 | 1.933 | 1.983 |
| `second_dist_x_blocked` | 0.2159 | -0.0716 | -0.0560 | 0.4066 | 1.933 | 1.983 |
| `action_4` | 0.4393 | 0.0200 | -0.0828 | 0.4247 | 0.12 | 0.2706 |
| `action_4_x_blocked` | 0.4393 | 0.0200 | -0.0828 | 0.4247 | 0.12 | 0.2706 |

### 3-block, H=72

| feature | std | corr(gain) | corr(route) | AUC(route) | mean route=1 | mean route=0 |
|---|---:|---:|---:|---:|---:|---:|
| `cheap_pred_margin` | 0.07814 | -0.1260 | -0.2935 | 0.2415 | 0.0333 | 0.09513 |
| `margin_x_blocked` | 0.07814 | -0.1260 | -0.2935 | 0.2415 | 0.0333 | 0.09513 |
| `cheap_confidence` | 0.1836 | -0.1600 | -0.2996 | 0.2455 | 0.7203 | 0.8686 |
| `cheap_uncertainty` | 0.1836 | 0.1600 | 0.2996 | 0.7545 | 0.2797 | 0.1314 |
| `uncertainty_x_blocked` | 0.1836 | 0.1600 | 0.2996 | 0.7545 | 0.2797 | 0.1314 |
| `uncertainty_x_horizon` | 0.1836 | 0.1600 | 0.2996 | 0.7545 | 0.2797 | 0.1314 |
| `npz_candidate_blocked_mean` | 0.1836 | -0.1530 | -0.2193 | 0.3325 | 0.4265 | 0.5351 |
| `cheap_second_best_distance` | 0.2382 | -0.1553 | -0.1724 | 0.3616 | 1.831 | 1.942 |
| `second_dist_x_blocked` | 0.2382 | -0.1553 | -0.1724 | 0.3616 | 1.831 | 1.942 |
| `npz_anchor_blocked` | 0.4446 | 0.0761 | 0.2268 | 0.6359 | 0.9559 | 0.6841 |
| `npz_candidate_blocked_std` | 0.06056 | 0.0825 | 0.1069 | 0.6346 | 0.4754 | 0.4579 |
| `npz_state_dist_max` | 0.0259 | 0.0305 | 0.0920 | 0.5870 | 0.123 | 0.1165 |
| `action_4` | 0.4254 | -0.0299 | -0.1249 | 0.4284 | 0.1176 | 0.2609 |
| `action_4_x_blocked` | 0.4254 | -0.0299 | -0.1249 | 0.4284 | 0.1176 | 0.2609 |
| `action_4_x_horizon` | 0.4254 | -0.0299 | -0.1249 | 0.4284 | 0.1176 | 0.2609 |

## Interpretation checklist

- If most added geometry features have near-zero std within a variant, they cannot help per-instance routing.
- If AUC(route) is close to 0.5, the feature is not predictive of oracle positive-gain routing.
- If context features dominate added geometry features, the v8 router should not be expected to beat v5.8.
- If some local geometry feature has strong AUC but the MLP fails, the next step is a simpler calibrated linear/tree router.

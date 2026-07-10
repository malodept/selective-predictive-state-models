# v62F crop-DINO action-coordinate ablation — v62e_crop_dino_features_n2000_h30_r80_crop320_seed0 seed=0

| action_mode | pca_dim | system | chance | oracle_pca | zero_delta | acc |
|---|---:|---|---:|---:|---:|---:|
| rel_block | 8 | `clean_action` | 0.200 | 1.000 | 0.200 | 0.699 |
| rel_block | 8 | `zero_action` | 0.200 | 1.000 | 0.200 | 0.200 |
| rel_block | 8 | `neg_action` | 0.200 | 1.000 | 0.200 | 0.129 |
| rel_block | 8 | `shuffled_action` | 0.200 | 1.000 | 0.200 | 0.083 |
| rel_block | 8 | `random_action` | 0.200 | 1.000 | 0.200 | 0.208 |
| rel_block | 8 | `pred_shuffled` | 0.200 | 1.000 | 0.200 | 0.123 |

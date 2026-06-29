# SPSM v33 OOD grid audit

This audit inventories existing OOD datasets and group files before expanding the controlled OOD grid.

## Existing NPZ inventory

- files found: `19`
- total size GB: `26.427`

| parent | file | MB | groups? | dinov2? |
|---|---|---:|---|---|
| block2_h72_seed11 | `mixed_hard_state_disjoint_v0_groups.npz` | 0.14 | True | False |
| block2_h72_seed11 | `pybullet_obstacle_rgb_v1_ood_block2_h72_seed11_dinov2_vits14.npz` | 3469.06 | False | True |
| block2_h72_seed11 | `pybullet_obstacle_rgb_v1_ood_block2_h72_seed11_dinov2_vits14_pool4.npz` | 139.00 | False | True |
| block2_v18_seed12 | `mixed_hard_state_disjoint_v0_groups.npz` | 0.14 | True | False |
| block2_v18_seed12 | `pybullet_obstacle_rgb_v1_ood_block2_v18_seed12_dinov2_vits14.npz` | 3468.99 | False | True |
| block2_v18_seed12 | `pybullet_obstacle_rgb_v1_ood_block2_v18_seed12_dinov2_vits14_pool4.npz` | 139.03 | False | True |
| block3_h36_seed13 | `mixed_hard_state_disjoint_v0_groups.npz` | 0.14 | True | False |
| block3_h36_seed13 | `pybullet_obstacle_rgb_v1_ood_block3_h36_seed13_dinov2_vits14.npz` | 3469.04 | False | True |
| block3_h36_seed13 | `pybullet_obstacle_rgb_v1_ood_block3_h36_seed13_dinov2_vits14_pool4.npz` | 139.03 | False | True |
| block3_h48_seed10 | `mixed_hard_state_disjoint_v0_groups.npz` | 0.14 | True | False |
| block3_h48_seed10 | `pybullet_obstacle_rgb_v1_ood_block3_h48_seed10_dinov2_vits14.npz` | 3468.93 | False | True |
| block3_h48_seed10 | `pybullet_obstacle_rgb_v1_ood_block3_h48_seed10_dinov2_vits14_pool4.npz` | 139.03 | False | True |
| block3_h72_seed14 | `mixed_hard_state_disjoint_v0_groups.npz` | 0.14 | True | False |
| block3_h72_seed14 | `pybullet_obstacle_rgb_v1_ood_block3_h72_seed14_dinov2_vits14.npz` | 3468.96 | False | True |
| block3_h72_seed14 | `pybullet_obstacle_rgb_v1_ood_block3_h72_seed14_dinov2_vits14_pool4.npz` | 139.03 | False | True |
| mixed_hard | `mixed_hard_v0_10k_groups.npz` | 0.41 | True | False |
| mixed_hard_state_disjoint | `mixed_hard_state_disjoint_v0_groups.npz` | 0.38 | True | False |
| pybullet_obstacles_rgb_scale | `pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14.npz` | 8672.28 | False | True |
| pybullet_obstacles_rgb_scale | `pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz` | 347.55 | False | True |

## Existing scalar metadata

| parent | file | horizon_value | velocity_value | num_blocked_directions_value | dataset_version_value | image_size_value |
|---|---|---|---|---|---|---|
| block2_h72_seed11 | `mixed_hard_state_disjoint_v0_groups.npz` |  |  |  |  |  |
| block2_h72_seed11 | `pybullet_obstacle_rgb_v1_ood_block2_h72_seed11_dinov2_vits14.npz` | 72 | 1.2 | 2 | pybullet_obstacle_exact_intervention_v1_rgb | 224 |
| block2_h72_seed11 | `pybullet_obstacle_rgb_v1_ood_block2_h72_seed11_dinov2_vits14_pool4.npz` | 72 | 1.2 | 2 | pybullet_obstacle_exact_intervention_v1_rgb | 224 |
| block2_v18_seed12 | `mixed_hard_state_disjoint_v0_groups.npz` |  |  |  |  |  |
| block2_v18_seed12 | `pybullet_obstacle_rgb_v1_ood_block2_v18_seed12_dinov2_vits14.npz` | 36 | 1.8 | 2 | pybullet_obstacle_exact_intervention_v1_rgb | 224 |
| block2_v18_seed12 | `pybullet_obstacle_rgb_v1_ood_block2_v18_seed12_dinov2_vits14_pool4.npz` | 36 | 1.8 | 2 | pybullet_obstacle_exact_intervention_v1_rgb | 224 |
| block3_h36_seed13 | `mixed_hard_state_disjoint_v0_groups.npz` |  |  |  |  |  |
| block3_h36_seed13 | `pybullet_obstacle_rgb_v1_ood_block3_h36_seed13_dinov2_vits14.npz` | 36 | 1.2 | 3 | pybullet_obstacle_exact_intervention_v1_rgb | 224 |
| block3_h36_seed13 | `pybullet_obstacle_rgb_v1_ood_block3_h36_seed13_dinov2_vits14_pool4.npz` | 36 | 1.2 | 3 | pybullet_obstacle_exact_intervention_v1_rgb | 224 |
| block3_h48_seed10 | `mixed_hard_state_disjoint_v0_groups.npz` |  |  |  |  |  |
| block3_h48_seed10 | `pybullet_obstacle_rgb_v1_ood_block3_h48_seed10_dinov2_vits14.npz` | 48 | 1.2 | 3 | pybullet_obstacle_exact_intervention_v1_rgb | 224 |
| block3_h48_seed10 | `pybullet_obstacle_rgb_v1_ood_block3_h48_seed10_dinov2_vits14_pool4.npz` | 48 | 1.2 | 3 | pybullet_obstacle_exact_intervention_v1_rgb | 224 |
| block3_h72_seed14 | `mixed_hard_state_disjoint_v0_groups.npz` |  |  |  |  |  |
| block3_h72_seed14 | `pybullet_obstacle_rgb_v1_ood_block3_h72_seed14_dinov2_vits14.npz` | 72 | 1.2 | 3 | pybullet_obstacle_exact_intervention_v1_rgb | 224 |
| block3_h72_seed14 | `pybullet_obstacle_rgb_v1_ood_block3_h72_seed14_dinov2_vits14_pool4.npz` | 72 | 1.2 | 3 | pybullet_obstacle_exact_intervention_v1_rgb | 224 |
| mixed_hard | `mixed_hard_v0_10k_groups.npz` |  |  |  |  |  |
| mixed_hard_state_disjoint | `mixed_hard_state_disjoint_v0_groups.npz` |  |  |  |  |  |
| pybullet_obstacles_rgb_scale | `pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14.npz` | 36 | 1.2 | 2 | pybullet_obstacle_exact_intervention_v1_rgb | 224 |
| pybullet_obstacles_rgb_scale | `pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz` | 36 | 1.2 | 2 | pybullet_obstacle_exact_intervention_v1_rgb | 224 |

## Proposed v33 goal

The next experiment should expand from a few named OOD variants to a systematic grid over geometry, horizon, velocity, and layout seed.
The first priority is to identify the correct dataset-generation script and its CLI flags from the code-context audit.

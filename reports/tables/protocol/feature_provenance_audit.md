# Feature provenance audit

## Source-code / text matches

## README.md
- L7: `The current benchmark uses TartanAir visual trajectories, frozen visual encoders, pose-conditioned latent prediction, and reliability-driven selective refinement.`
- L53: `- **TartanAir JapaneseAlley-Hard**`
- L54: `- 6 trajectories: `P000` to `P005``
- L56: `- RGB frames and camera poses`
- L57: `- action representation: camera-pose difference between frame \(t\) and frame \(t+k\)`
- L59: `The action is not a motor command in the current benchmark. It is a pose-difference descriptor used to condition latent transition prediction.`
- L82: `A transition is labeled difficult if the temporal gap or pose displacement is large.`
- L199: `evaluate transfer beyond TartanAir.`

## configs/data/tartanair_base.yaml
- L1: `name: tartanair`
- L2: `root: data/raw/tartanair`

## configs/train/image_sequence_feature_smoke.yaml
- L6: `  train_path: outputs/image_sequence_features/features_train.npz`
- L7: `  val_path: outputs/image_sequence_features/features_val.npz`

## configs/train/image_sequence_resnet18_smoke.yaml
- L6: `  train_path: outputs/image_sequence_resnet18_features/features_train.npz`
- L7: `  val_path: outputs/image_sequence_resnet18_features/features_val.npz`

## configs/train/real_video_resnet18_smoke.yaml
- L6: `  train_path: outputs/real_video_resnet18_features/features_train.npz`
- L7: `  val_path: outputs/real_video_resnet18_features/features_val.npz`

## configs/train/tartanair_dinov2_error_reliability.yaml
- L2: `output_dir: outputs/tartanair_dinov2_error_reliability`
- L6: `  train_path: outputs/tartanair_dinov2_features/features_train_errorrel.npz`
- L7: `  val_path: outputs/tartanair_dinov2_features/features_val_errorrel.npz`

## configs/train/tartanair_dinov2_error_reliability_seed0.yaml
- L2: `output_dir: outputs/tartanair_dinov2_error_reliability_seed0`
- L6: `  train_path: outputs/tartanair_dinov2_features/features_train_errorrel.npz`
- L7: `  val_path: outputs/tartanair_dinov2_features/features_val_errorrel.npz`

## configs/train/tartanair_dinov2_error_reliability_seed1.yaml
- L2: `output_dir: outputs/tartanair_dinov2_error_reliability_seed1`
- L6: `  train_path: outputs/tartanair_dinov2_features/features_train_errorrel.npz`
- L7: `  val_path: outputs/tartanair_dinov2_features/features_val_errorrel.npz`

## configs/train/tartanair_dinov2_error_reliability_seed2.yaml
- L2: `output_dir: outputs/tartanair_dinov2_error_reliability_seed2`
- L6: `  train_path: outputs/tartanair_dinov2_features/features_train_errorrel.npz`
- L7: `  val_path: outputs/tartanair_dinov2_features/features_val_errorrel.npz`

## configs/train/tartanair_dinov2_full.yaml
- L2: `output_dir: outputs/tartanair_dinov2_full`
- L6: `  train_path: outputs/tartanair_dinov2_features/features_train.npz`
- L7: `  val_path: outputs/tartanair_dinov2_features/features_val.npz`

## configs/train/tartanair_dinov2_l2_error_reliability.yaml
- L2: `output_dir: outputs/tartanair_dinov2_l2_error_reliability`
- L6: `  train_path: outputs/tartanair_dinov2_features/features_train_l2_errorrel.npz`
- L7: `  val_path: outputs/tartanair_dinov2_features/features_val_l2_errorrel.npz`

## configs/train/tartanair_dinov2_l2_full.yaml
- L2: `output_dir: outputs/tartanair_dinov2_l2_full`
- L6: `  train_path: outputs/tartanair_dinov2_features/features_train_l2.npz`
- L7: `  val_path: outputs/tartanair_dinov2_features/features_val_l2.npz`

## configs/train/tartanair_resnet18_error_reliability.yaml
- L2: `output_dir: outputs/tartanair_resnet18_error_reliability`
- L6: `  train_path: outputs/tartanair_resnet18_features_full/features_train_errorrel.npz`
- L7: `  val_path: outputs/tartanair_resnet18_features_full/features_val_errorrel.npz`

## configs/train/tartanair_resnet18_error_reliability_seed0.yaml
- L2: `output_dir: outputs/tartanair_resnet18_error_reliability_seed0`
- L6: `  train_path: outputs/tartanair_resnet18_features_full/features_train_errorrel.npz`
- L7: `  val_path: outputs/tartanair_resnet18_features_full/features_val_errorrel.npz`

## configs/train/tartanair_resnet18_error_reliability_seed1.yaml
- L2: `output_dir: outputs/tartanair_resnet18_error_reliability_seed1`
- L6: `  train_path: outputs/tartanair_resnet18_features_full/features_train_errorrel.npz`
- L7: `  val_path: outputs/tartanair_resnet18_features_full/features_val_errorrel.npz`

## configs/train/tartanair_resnet18_error_reliability_seed2.yaml
- L2: `output_dir: outputs/tartanair_resnet18_error_reliability_seed2`
- L6: `  train_path: outputs/tartanair_resnet18_features_full/features_train_errorrel.npz`
- L7: `  val_path: outputs/tartanair_resnet18_features_full/features_val_errorrel.npz`

## configs/train/tartanair_resnet18_full.yaml
- L2: `output_dir: outputs/tartanair_resnet18_full`
- L6: `  train_path: outputs/tartanair_resnet18_features_full/features_train.npz`
- L7: `  val_path: outputs/tartanair_resnet18_features_full/features_val.npz`

## configs/train/tartanair_resnet18_tiny.yaml
- L2: `output_dir: outputs/tartanair_resnet18_tiny`
- L6: `  train_path: outputs/tartanair_resnet18_features/features_train.npz`
- L7: `  val_path: outputs/tartanair_resnet18_features/features_val.npz`

## data/prepare_tartanair.py
- L1: `"""Placeholder for public controlled trajectory data preparation."""`

## docs/datasets.md
- L7: `## Public controlled trajectory data`
- L18: ``scripts/extract_frames_from_videos.py` extracts fixed-rate frame sequences into `data/real_video_sequences/`.`

## docs/experiments.md
- L5: `Purpose: prove that the repo trains, evaluates retrieval, estimates surprise, and exports a compute-utility curve.`
- L22: `Purpose: compare static cheap, static expensive, and adaptive policies.`
- L31: `Purpose: replace synthetic latents with public trajectory data and frozen teacher features.`

## docs/roadmap.md
- L13: `- Public trajectory loader skeleton.`

## outputs/bestval_dinov2_cheap10_exp120_seed0/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap10_exp120_seed1/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap10_exp120_seed2/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap10_exp40_seed0/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap10_exp40_seed1/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap10_exp40_seed2/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap10_exp80_seed0/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap10_exp80_seed1/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap10_exp80_seed2/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap20_exp80_seed0/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap20_exp80_seed1/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap20_exp80_seed2/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap5_exp80_seed0/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap5_exp80_seed1/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## outputs/bestval_dinov2_cheap5_exp80_seed2/metrics.json
- L3: `  "train": "outputs/tartanair_dinov2_features/features_train.npz",`
- L4: `  "val": "outputs/tartanair_dinov2_features/features_val.npz",`

## releases/spsm_v1_0_rc1/README.md
- L11: `- Dataset: TartanAir JapaneseAlley-Hard`
- L12: `- Trajectories: P000 to P005`
- L14: `- Input: RGB frames and camera poses`
- L15: `- Action representation: pose difference between frame t and frame t+k`

## releases/spsm_v1_0_rc1/configs/tartanair_dinov2_error_reliability.yaml
- L2: `output_dir: outputs/tartanair_dinov2_error_reliability`
- L6: `  train_path: outputs/tartanair_dinov2_features/features_train_errorrel.npz`
- L7: `  val_path: outputs/tartanair_dinov2_features/features_val_errorrel.npz`

## releases/spsm_v1_0_rc1/configs/tartanair_dinov2_full.yaml
- L2: `output_dir: outputs/tartanair_dinov2_full`
- L6: `  train_path: outputs/tartanair_dinov2_features/features_train.npz`
- L7: `  val_path: outputs/tartanair_dinov2_features/features_val.npz`

## releases/spsm_v1_0_rc1/configs/tartanair_resnet18_error_reliability.yaml
- L2: `output_dir: outputs/tartanair_resnet18_error_reliability`
- L6: `  train_path: outputs/tartanair_resnet18_features_full/features_train_errorrel.npz`
- L7: `  val_path: outputs/tartanair_resnet18_features_full/features_val_errorrel.npz`

## releases/spsm_v1_0_rc1/configs/tartanair_resnet18_full.yaml
- L2: `output_dir: outputs/tartanair_resnet18_full`
- L6: `  train_path: outputs/tartanair_resnet18_features_full/features_train.npz`
- L7: `  val_path: outputs/tartanair_resnet18_features_full/features_val.npz`

## releases/spsm_v1_0_rc1/scripts/reproduce_main_runs.sh
- L12: `ls configs/train/tartanair_resnet18_full.yaml`
- L13: `ls configs/train/tartanair_resnet18_error_reliability.yaml`
- L14: `ls configs/train/tartanair_dinov2_full.yaml`
- L15: `ls configs/train/tartanair_dinov2_error_reliability.yaml`
- L18: `ls outputs/tartanair_resnet18_full/metrics.json`
- L19: `ls outputs/tartanair_resnet18_error_reliability/metrics.json`
- L20: `ls outputs/tartanair_dinov2_full/metrics.json`
- L21: `ls outputs/tartanair_dinov2_error_reliability/metrics.json`

## reports/real_video_smoke_note.md
- L3: `This experiment replaces generated image sequences with real video frames while keeping the same SPSM pipeline:`
- L6: `video files -> extracted frames -> frozen ResNet18 features -> predictive latent model -> reliability -> selective compute diagnostics`
- L9: `The purpose is not to claim final performance. It checks whether the SPSM training and evaluation pipeline remains valid when the latent states are extracted from real temporal observations.`

## reports/synthetic_selective_compute_note.md
- L3: `## Purpose`

## reports/tables/protocol/data_layout_audit.md
- L78: `\| `outputs/real_video_resnet18_features/features.npz` \| 91.24 \| npz unreadable: ValueError: Object arrays cannot be loaded when allow_pickle=False \|`
- L79: `\| `outputs/real_video_resnet18_features/features_train.npz` \| 60.87 \| npz unreadable: ValueError: Object arrays cannot be loaded when allow_pickle=False \|`
- L80: `\| `outputs/real_video_resnet18_features/features_val.npz` \| 20.28 \| npz unreadable: ValueError: Object arrays cannot be loaded when allow_pickle=False \|`
- L85: `\| `outputs/tartanair_dinov2_error_reliability/checkpoint.pt` \| 1.79 \| binary checkpoint/cache; not loaded for safety \|`
- L86: `\| `outputs/tartanair_dinov2_error_reliability/eval_arrays.npz` \| 0.06 \| npz probs: shape=(3292,), dtype=float32; residuals_observed: shape=(3292,), dtype=float32; residuals_clean: shape=(3292,), dtype=float32; expecte`
- L87: `\| `outputs/tartanair_dinov2_error_reliability/metrics.json` \| 0.01 \| json dict keys=['history', 'latency', 'parameter_count', 'retrieval', 'selector_best_utility', 'selector_rows', 'surprise'], useful_keys=[] \|`
- L88: `\| `outputs/tartanair_dinov2_error_reliability/selector_utility.csv` \| 0.00 \| csv cols=6, useful_cols=['mean_error', 'selected_fraction'] \|`
- L89: `\| `outputs/tartanair_dinov2_error_reliability_seed0/checkpoint.pt` \| 1.79 \| binary checkpoint/cache; not loaded for safety \|`
- L90: `\| `outputs/tartanair_dinov2_error_reliability_seed0/eval_arrays.npz` \| 0.06 \| npz probs: shape=(3292,), dtype=float32; residuals_observed: shape=(3292,), dtype=float32; residuals_clean: shape=(3292,), dtype=float32; e`
- L91: `\| `outputs/tartanair_dinov2_error_reliability_seed0/metrics.json` \| 0.01 \| json dict keys=['history', 'latency', 'parameter_count', 'retrieval', 'selector_best_utility', 'selector_rows', 'surprise'], useful_keys=[] \|`
- L92: `\| `outputs/tartanair_dinov2_error_reliability_seed0/selector_utility.csv` \| 0.00 \| csv cols=6, useful_cols=['mean_error', 'selected_fraction'] \|`
- L93: `\| `outputs/tartanair_dinov2_error_reliability_seed1/checkpoint.pt` \| 1.79 \| binary checkpoint/cache; not loaded for safety \|`
- L94: `\| `outputs/tartanair_dinov2_error_reliability_seed1/eval_arrays.npz` \| 0.06 \| npz probs: shape=(3292,), dtype=float32; residuals_observed: shape=(3292,), dtype=float32; residuals_clean: shape=(3292,), dtype=float32; e`
- L95: `\| `outputs/tartanair_dinov2_error_reliability_seed1/metrics.json` \| 0.01 \| json dict keys=['history', 'latency', 'parameter_count', 'retrieval', 'selector_best_utility', 'selector_rows', 'surprise'], useful_keys=[] \|`
- L96: `\| `outputs/tartanair_dinov2_error_reliability_seed1/selector_utility.csv` \| 0.00 \| csv cols=6, useful_cols=['mean_error', 'selected_fraction'] \|`
- L97: `\| `outputs/tartanair_dinov2_error_reliability_seed2/checkpoint.pt` \| 1.79 \| binary checkpoint/cache; not loaded for safety \|`
- L98: `\| `outputs/tartanair_dinov2_error_reliability_seed2/eval_arrays.npz` \| 0.06 \| npz probs: shape=(3292,), dtype=float32; residuals_observed: shape=(3292,), dtype=float32; residuals_clean: shape=(3292,), dtype=float32; e`
- L99: `\| `outputs/tartanair_dinov2_error_reliability_seed2/metrics.json` \| 0.01 \| json dict keys=['history', 'latency', 'parameter_count', 'retrieval', 'selector_best_utility', 'selector_rows', 'surprise'], useful_keys=[] \|`
- L100: `\| `outputs/tartanair_dinov2_error_reliability_seed2/selector_utility.csv` \| 0.00 \| csv cols=6, useful_cols=['mean_error', 'selected_fraction'] \|`
- L101: `\| `outputs/tartanair_dinov2_features/error_reliability_l2_label_report.json` \| 0.00 \| json dict keys=['latent_dim', 'action_dim', 'hard_fraction', 'threshold', 'train_error_mean', 'train_error_std', 'val_error_mean', `
- L102: `\| `outputs/tartanair_dinov2_features/error_reliability_label_report.json` \| 0.00 \| json dict keys=['latent_dim', 'action_dim', 'hard_fraction', 'threshold', 'train_error_mean', 'train_error_std', 'val_error_mean', 'va`
- L103: `\| `outputs/tartanair_dinov2_features/features.npz` \| 10.41 \| npz z_current: shape=(13170, 384), dtype=float32; z_future: shape=(13170, 384), dtype=float32; action: shape=(13170, 7), dtype=float32; expected_unreliable:`
- L104: `\| `outputs/tartanair_dinov2_features/features_train.npz` \| 26.95 \| npz z_current: shape=(9878, 384), dtype=float32; z_future: shape=(9878, 384), dtype=float32; action: shape=(9878, 7), dtype=float32; expected_unreliab`
- L105: `\| `outputs/tartanair_dinov2_features/features_train_errorrel.npz` \| 26.99 \| npz z_current: shape=(9878, 384), dtype=float32; z_future: shape=(9878, 384), dtype=float32; action: shape=(9878, 7), dtype=float32; expected`
- L106: `\| `outputs/tartanair_dinov2_features/features_train_l2.npz` \| 26.89 \| npz z_current: shape=(9878, 384), dtype=float32; z_future: shape=(9878, 384), dtype=float32; action: shape=(9878, 7), dtype=float32; expected_unrel`
- L107: `\| `outputs/tartanair_dinov2_features/features_train_l2_errorrel.npz` \| 26.92 \| npz z_current: shape=(9878, 384), dtype=float32; z_future: shape=(9878, 384), dtype=float32; action: shape=(9878, 7), dtype=float32; expec`
- L108: `\| `outputs/tartanair_dinov2_features/features_val.npz` \| 8.98 \| npz z_current: shape=(3292, 384), dtype=float32; z_future: shape=(3292, 384), dtype=float32; action: shape=(3292, 7), dtype=float32; expected_unreliable:`
- L109: `\| `outputs/tartanair_dinov2_features/features_val_errorrel.npz` \| 8.99 \| npz z_current: shape=(3292, 384), dtype=float32; z_future: shape=(3292, 384), dtype=float32; action: shape=(3292, 7), dtype=float32; expected_un`
- L110: `\| `outputs/tartanair_dinov2_features/features_val_l2.npz` \| 8.96 \| npz z_current: shape=(3292, 384), dtype=float32; z_future: shape=(3292, 384), dtype=float32; action: shape=(3292, 7), dtype=float32; expected_unreliab`
- L111: `\| `outputs/tartanair_dinov2_features/features_val_l2_errorrel.npz` \| 8.97 \| npz z_current: shape=(3292, 384), dtype=float32; z_future: shape=(3292, 384), dtype=float32; action: shape=(3292, 7), dtype=float32; expected`
- ... 103 more hits

## reports/tables/protocol/feature_provenance_audit.md
- L6: `- L7: `The current benchmark uses TartanAir visual trajectories, frozen visual encoders, pose-conditioned latent prediction, and reliability-driven selective refinement.``
- L7: `- L53: `- **TartanAir JapaneseAlley-Hard**``
- L8: `- L54: `- 6 trajectories: `P000` to `P005```
- L9: `- L56: `- RGB frames and camera poses``
- L10: `- L57: `- action representation: camera-pose difference between frame \(t\) and frame \(t+k\)``
- L11: `- L59: `The action is not a motor command in the current benchmark. It is a pose-difference descriptor used to condition latent transition prediction.``
- L12: `- L82: `A transition is labeled difficult if the temporal gap or pose displacement is large.``
- L13: `- L199: `evaluate transfer beyond TartanAir.``
- L15: `## configs/data/tartanair_base.yaml`
- L16: `- L1: `name: tartanair``
- L17: `- L2: `root: data/raw/tartanair``
- L20: `- L6: `  train_path: outputs/image_sequence_features/features_train.npz``
- L21: `- L7: `  val_path: outputs/image_sequence_features/features_val.npz``
- L24: `- L6: `  train_path: outputs/image_sequence_resnet18_features/features_train.npz``
- L25: `- L7: `  val_path: outputs/image_sequence_resnet18_features/features_val.npz``
- L28: `- L6: `  train_path: outputs/real_video_resnet18_features/features_train.npz``
- L29: `- L7: `  val_path: outputs/real_video_resnet18_features/features_val.npz``
- L31: `## configs/train/tartanair_dinov2_error_reliability.yaml`
- L32: `- L2: `output_dir: outputs/tartanair_dinov2_error_reliability``
- L33: `- L6: `  train_path: outputs/tartanair_dinov2_features/features_train_errorrel.npz``
- L34: `- L7: `  val_path: outputs/tartanair_dinov2_features/features_val_errorrel.npz``
- L36: `## configs/train/tartanair_dinov2_error_reliability_seed0.yaml`
- L37: `- L2: `output_dir: outputs/tartanair_dinov2_error_reliability_seed0``
- L38: `- L6: `  train_path: outputs/tartanair_dinov2_features/features_train_errorrel.npz``
- L39: `- L7: `  val_path: outputs/tartanair_dinov2_features/features_val_errorrel.npz``
- L41: `## configs/train/tartanair_dinov2_error_reliability_seed1.yaml`
- L42: `- L2: `output_dir: outputs/tartanair_dinov2_error_reliability_seed1``
- L43: `- L6: `  train_path: outputs/tartanair_dinov2_features/features_train_errorrel.npz``
- L44: `- L7: `  val_path: outputs/tartanair_dinov2_features/features_val_errorrel.npz``
- L46: `## configs/train/tartanair_dinov2_error_reliability_seed2.yaml`
- ... 133 more hits

## reports/visual_feature_smoke_note.md
- L13: `The default extractor is a dependency-light patch-mean encoder, not a final research encoder. It exists to validate the data flow and evaluation protocol before plugging in DINO/V-JEPA/TartanAir features.`

## reports/visual_resnet18_smoke_note.md
- L3: `This experiment replaces the dependency-light patch-mean encoder with a frozen ResNet18 visual feature extractor. The purpose is not to claim a final visual world model, but to verify that the SPSM pipeline can operate o`
- L8: `2. Encode each frame with frozen ResNet18.`
- L12: `This is the bridge between the synthetic latent proof of concept and future public datasets such as TartanAir, TartanDrive, DROID, or other trajectory datasets.`

## scripts/aggregate_dinov2_errorrel_seeds.py
- L8: `    Path("outputs/tartanair_dinov2_error_reliability_seed0"),`
- L9: `    Path("outputs/tartanair_dinov2_error_reliability_seed1"),`
- L10: `    Path("outputs/tartanair_dinov2_error_reliability_seed2"),`

## scripts/aggregate_errorrel_seeds.py
- L10: `    Path("outputs/tartanair_resnet18_error_reliability_seed0"),`
- L11: `    Path("outputs/tartanair_resnet18_error_reliability_seed1"),`
- L12: `    Path("outputs/tartanair_resnet18_error_reliability_seed2"),`

## scripts/audit_feature_provenance.py
- L10: `    "tartanair_dinov2_features",`
- L11: `    "features_train.npz",`
- L12: `    "features_val.npz",`
- L13: `    "features.npz",`
- L14: `    "z_current",`
- L15: `    "z_future",`
- L16: `    "np.savez",`
- L17: `    "savez",`
- L18: `    "train_test_split",`
- L19: `    "random_split",`
- L20: `    "shuffle",`
- L21: `    "TartanAir",`
- L22: `    "tartanair",`
- L23: `    "JapaneseAlley",`
- L24: `    "trajectory",`
- L25: `    "traj",`
- L26: `    "frame",`
- L27: `    "pose",`
- L28: `    "image_left",`
- L41: `    ROOT / "outputs/tartanair_dinov2_features/features.npz",`
- L42: `    ROOT / "outputs/tartanair_dinov2_features/features_train.npz",`
- L43: `    ROOT / "outputs/tartanair_dinov2_features/features_val.npz",`
- L44: `    ROOT / "outputs/tartanair_dinov2_features/features_train_errorrel.npz",`
- L45: `    ROOT / "outputs/tartanair_dinov2_features/features_val_errorrel.npz",`
- L46: `    ROOT / "outputs/tartanair_resnet18_features_full/features.npz",`
- L47: `    ROOT / "outputs/tartanair_resnet18_features_full/features_train.npz",`
- L48: `    ROOT / "outputs/tartanair_resnet18_features_full/features_val.npz",`
- L144: `        if any(s in low for s in ["traj", "trajectory", "seq", "sequence", "frame", "idx", "index", "path", "file", "scene", "env"]):`

## scripts/audit_protocol_inputs.py
- L29: `    "traj",`
- L30: `    "trajectory",`
- L33: `    "frame",`
- L38: `    "pose",`

## scripts/compare_tartanair_runs.py
- L8: `    "v0.7_full_heuristic": Path("outputs/tartanair_resnet18_full/metrics.json"),`
- L9: `    "v0.8_error_reliability": Path("outputs/tartanair_resnet18_error_reliability/metrics.json"),`
- L48: `    out = Path("reports/tables/tartanair_v07_v08_comparison.md")`

## scripts/export_bestval_eval_arrays.py
- L101: `    np.savez_compressed(`

## scripts/extract_frames_from_videos.py
- L17: `        description="Extract frame sequences from videos for SPSM real-video smoke tests."`
- L22: `    parser.add_argument("--max-frames-per-video", type=int, default=120)`
- L29: `        help="Frame extraction backend. 'auto' tries OpenCV first, then ffmpeg CLI.",`
- L44: `def resize_and_save(frame_rgb, path: Path, image_size: int) -> None:`
- L45: `    image = Image.fromarray(frame_rgb)`
- L50: `def extract_with_opencv(video_path: Path, seq_dir: Path, fps: float, max_frames: int, image_size: int) -> int:`
- L66: `    frame_idx = 0`
- L67: `    while written < max_frames:`
- L68: `        ok, frame_bgr = cap.read()`
- L71: `        if frame_idx % stride == 0:`
- L72: `            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)`
- L73: `            resize_and_save(frame_rgb, seq_dir / f"{written:04d}.png", image_size)`
- L75: `        frame_idx += 1`
- L81: `def extract_with_ffmpeg(video_path: Path, seq_dir: Path, fps: float, max_frames: int, image_size: int) -> int:`
- L97: `        "-frames:v",`
- L98: `        str(max_frames),`
- L109: `    # Avoid mixing old and new frames when overwrite is false.`
- L116: `                video_path, seq_dir, args.fps, args.max_frames_per_video, args.image_size`
- L122: `    return extract_with_ffmpeg(video_path, seq_dir, args.fps, args.max_frames_per_video, args.image_size)`
- L138: `    for seq_id, video in enumerate(tqdm(videos, desc="extracting video frames")):`
- L141: `        print(f"{video.name}: wrote {n} frames to {args.output / f'seq_{seq_id:04d}'}")`
- L143: `    print(f"Done. Extracted {total} frames from {len(videos)} videos into {args.output}")`

## scripts/extract_image_sequence_features.py
- L12: `def sorted_frames(seq_dir: Path) -> list[Path]:`
- L17: `def build_sequence_features(frames: list[Path], args: argparse.Namespace) -> list[np.ndarray]:`
- L19: `        return [patch_mean_features(p, grid_size=args.grid_size) for p in frames]`
- L36: `        return list(extractor.encode_paths(frames))`
- L44: `    parser.add_argument("--output", default="outputs/image_sequence_features/features.npz")`
- L65: `    z_current, clean_future, actions, expected = [], [], [], []`
- L69: `        frames = sorted_frames(seq_dir)`
- L70: `        if len(frames) < 2:`
- L72: `        feats = build_sequence_features(frames, args)`
- L78: `                z_current.append(feats[i])`
- L84: `    z_current = np.stack(z_current).astype(np.float32)`
- L89: `    n = z_current.shape[0]`
- L92: `    z_future = clean_future.copy()`
- L93: `    z_future[observed_surprise] = clean_future[perm[observed_surprise]]`
- L97: `    np.savez(`
- L99: `        z_current=z_current,`
- L101: `        z_future=z_future.astype(np.float32),`
- L110: `    print(f"encoder={args.encoder} latent_dim={z_current.shape[1]} action_dim={action.shape[1]}")`

## scripts/extract_tartanair_dinov2_features.py
- L13: `IMAGE_DIR_CANDIDATES = ["image_left", "image_right"]`
- L14: `POSE_FILE_CANDIDATES = ["pose_left.txt", "pose_right.txt"]`
- L26: `    p.add_argument("--max-trajectories", type=int, default=6)`
- L27: `    p.add_argument("--max-frames-per-trajectory", type=int, default=600)`
- L33: `def find_trajectories(root: Path) -> list[tuple[Path, Path, Path]]:`
- L35: `    for traj_dir in root.rglob("*"):`
- L36: `        if not traj_dir.is_dir():`
- L41: `            c = traj_dir / name`
- L49: `        pose_file = None`
- L50: `        for name in POSE_FILE_CANDIDATES:`
- L51: `            c = traj_dir / name`
- L53: `                pose_file = c`
- L56: `        if pose_file is not None:`
- L57: `            found.append((traj_dir, image_dir, pose_file))`
- L66: `def load_poses(path: Path) -> np.ndarray:`
- L67: `    poses = np.loadtxt(path, dtype=np.float32)`
- L68: `    if poses.ndim == 1:`
- L69: `        poses = poses[None, :]`
- L70: `    return poses`
- L73: `def preprocess() -> transforms.Compose:`
- L74: `    return transforms.Compose(`
- L90: `    tfm: transforms.Compose,`
- L117: `    poses: np.ndarray,`
- L123: `    n = min(len(features), len(poses))`
- L125: `    poses = poses[:n]`
- L127: `    z_current, z_future, actions = [], [], []`
- L137: `            action = (poses[j] - poses[i]).astype(np.float32)`
- L148: `            z_current.append(z_i)`
- L149: `            z_future.append(z_j)`
- L155: `        "z_current": np.stack(z_current).astype(np.float32),`
- ... 17 more hits

## scripts/extract_tartanair_features.py
- L15: `    "image_left",`
- L22: `POSE_FILE_CANDIDATES = [`
- L23: `    "pose_left.txt",`
- L24: `    "pose_right.txt",`
- L25: `    "pose_lcam_front.txt",`
- L26: `    "pose_rcam_front.txt",`
- L27: `    "pose.txt",`
- L41: `    parser.add_argument("--max-trajectories", type=int, default=2)`
- L42: `    parser.add_argument("--max-frames-per-trajectory", type=int, default=300)`
- L59: `def preprocess() -> transforms.Compose:`
- L60: `    return transforms.Compose(`
- L72: `def find_trajectories(root: Path) -> list[tuple[Path, Path, Path]]:`
- L75: `    for traj_dir in root.rglob("*"):`
- L76: `        if not traj_dir.is_dir():`
- L81: `            candidate = traj_dir / name`
- L93: `        pose_file = None`
- L94: `        for name in POSE_FILE_CANDIDATES:`
- L95: `            candidate = traj_dir / name`
- L97: `                pose_file = candidate`
- L100: `        if pose_file is None:`
- L103: `        found.append((traj_dir, image_dir, pose_file))`
- L116: `def load_poses(path: Path) -> np.ndarray:`
- L117: `    poses = np.loadtxt(path, dtype=np.float32)`
- L118: `    if poses.ndim == 1:`
- L119: `        poses = poses[None, :]`
- L120: `    return poses`
- L127: `    tfm: transforms.Compose,`
- L133: `    for start in tqdm(range(0, len(images), batch_size), desc="encoding frames", leave=False):`
- L150: `    poses: np.ndarray,`
- L156: `    n = min(len(features), len(poses))`
- ... 28 more hits

## scripts/make_error_reliability_labels.py
- L63: `    z = tensor(data["z_current"], device)`
- L65: `    y = tensor(data["z_future"], device)`
- L67: `    loader = DataLoader(TensorDataset(z, a, y), batch_size=batch_size, shuffle=False)`
- L99: `    np.savez_compressed(output, **out)`
- L114: `    latent_dim = int(train["z_current"].shape[1])`
- L119: `    z_train = tensor(train["z_current"], args.device)`
- L121: `    y_train = tensor(train["z_future"], args.device)`
- L126: `        shuffle=True,`

## scripts/make_procedural_video_sequences.py
- L15: `    parser.add_argument("--n-frames", type=int, default=80)`
- L49: `def crop_with_motion(bg: Image.Image, t: int, n_frames: int, size: int, seq_id: int) -> Image.Image:`
- L53: `    phase = 2.0 * math.pi * t / max(n_frames - 1, 1)`
- L60: `    frame = bg.crop((x, y, x + size, y + size))`
- L64: `    frame = frame.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(20, 20, 20))`
- L66: `    return frame`
- L70: `    frame: Image.Image,`
- L73: `    n_frames: int,`
- L76: `    draw = ImageDraw.Draw(frame, "RGBA")`
- L77: `    w, h = frame.size`
- L80: `    phase = t / max(n_frames - 1, 1)`
- L93: `    arr = np.asarray(frame).astype(np.float32)`
- L106: `    for t in range(args.n_frames):`
- L107: `        frame = crop_with_motion(bg, t, args.n_frames, args.size, seq_id)`
- L108: `        frame = add_moving_objects(frame, rng, t, args.n_frames, seq_id)`
- L109: `        frame.save(seq_dir / f"{t:04d}.png")`

## scripts/make_sample_image_sequences.py
- L9: `def make_sequence(out_dir: Path, seq_id: int, n_frames: int, size: int) -> None:`
- L16: `    for t in range(n_frames):`
- L32: `    parser.add_argument("--n-frames", type=int, default=32)`
- L39: `        make_sequence(out_dir, seq_id, args.n_frames, args.size)`

## scripts/normalize_feature_npz.py
- L28: `    for key in ["z_current", "z_future"]:`
- L36: `    np.savez_compressed(args.output, **out)`
- L39: `    print("z_current norm mean:", np.linalg.norm(out["z_current"], axis=1).mean())`
- L40: `    print("z_future norm mean:", np.linalg.norm(out["z_future"], axis=1).mean())`

## scripts/rescore_dinov2_selector_utility.py
- L7: `    "dinov2_full_heuristic": Path("outputs/tartanair_dinov2_full/metrics.json"),`
- L8: `    "dinov2_error_reliability": Path("outputs/tartanair_dinov2_error_reliability/metrics.json"),`

## scripts/rescore_selector_utility.py
- L6: `RUN = Path("outputs/tartanair_resnet18_tiny/metrics.json")`

## scripts/run_bestval_gap_ablation.sh
- L5: `TRAIN="outputs/tartanair_dinov2_features/features_train.npz"`
- L6: `VAL="outputs/tartanair_dinov2_features/features_val.npz"`

## scripts/run_image_sequence_resnet18_smoke.sh
- L3: `python scripts/make_sample_image_sequences.py --output data/sample_image_sequences --n-sequences 12 --n-frames 16`
- L4: `python scripts/extract_image_sequence_features.py --input data/sample_image_sequences --output outputs/image_sequence_resnet18_features/features.npz --encoder resnet18 --resnet-weights imagenet --max-gap 3 --hard-gap 2`
- L5: `python scripts/split_feature_npz.py --input outputs/image_sequence_resnet18_features/features.npz`

## scripts/run_image_sequence_smoke.sh
- L3: `python scripts/make_sample_image_sequences.py --output data/sample_image_sequences --n-sequences 8 --n-frames 12`
- L4: `python scripts/extract_image_sequence_features.py --input data/sample_image_sequences --output outputs/image_sequence_features/features.npz --max-gap 3 --hard-gap 2`
- L5: `python scripts/split_feature_npz.py --input outputs/image_sequence_features/features.npz`

## scripts/run_real_video_resnet18_smoke.sh
- L4: `python scripts/extract_frames_from_videos.py \`
- L8: `  --max-frames-per-video 120 \`
- L14: `  --output outputs/real_video_resnet18_features/features.npz \`
- L22: `  --input outputs/real_video_resnet18_features/features.npz`

## scripts/split_feature_npz.py
- L21: `        "z_current",`
- L22: `        "z_future",`
- L53: `    rng.shuffle(indices)`
- L69: `        np.savez_compressed(path, **out)`

## scripts/teacher_ablation_main_table.py
- L7: `    "ResNet18 heuristic": Path("outputs/tartanair_resnet18_full/metrics.json"),`
- L8: `    "ResNet18 error-rel": Path("outputs/tartanair_resnet18_error_reliability/metrics.json"),`
- L9: `    "DINOv2 heuristic": Path("outputs/tartanair_dinov2_full/metrics.json"),`
- L10: `    "DINOv2 error-rel": Path("outputs/tartanair_dinov2_error_reliability/metrics.json"),`

## scripts/teacher_ablation_table.py
- L7: `    "ResNet18 heuristic": Path("outputs/tartanair_resnet18_full/metrics.json"),`
- L8: `    "ResNet18 error-rel": Path("outputs/tartanair_resnet18_error_reliability/metrics.json"),`
- L9: `    "DINOv2 heuristic": Path("outputs/tartanair_dinov2_full/metrics.json"),`
- L10: `    "DINOv2 error-rel": Path("outputs/tartanair_dinov2_error_reliability/metrics.json"),`
- L11: `    "DINOv2 L2 heuristic": Path("outputs/tartanair_dinov2_l2_full/metrics.json"),`
- L12: `    "DINOv2 L2 error-rel": Path("outputs/tartanair_dinov2_l2_error_reliability/metrics.json"),`

## scripts/train_real_selective_refinement.py
- L89: `def make_loader(data: dict[str, np.ndarray], device: str, batch_size: int, shuffle: bool) -> DataLoader:`
- L90: `    z = as_tensor(data["z_current"], device)`
- L92: `    y = as_tensor(data["z_future"], device)`
- L93: `    return DataLoader(TensorDataset(z, a, y), batch_size=batch_size, shuffle=shuffle)`
- L149: `    z = as_tensor(train_data["z_current"], args.device)`
- L151: `    y = as_tensor(train_data["z_future"], args.device)`
- L158: `    loader = DataLoader(TensorDataset(z, a, zhat.detach(), target), batch_size=args.batch_size, shuffle=True)`
- L286: `    latent_dim = int(train["z_current"].shape[1])`
- L289: `    train_loader = make_loader(train, args.device, args.batch_size, shuffle=True)`
- L290: `    val_loader = make_loader(val, args.device, args.batch_size, shuffle=False)`

## scripts/train_real_selective_refinement_bestval.py
- L27: `    if "z_current" not in data or "z_future" not in data:`
- L28: `        raise KeyError(f"{path} must contain z_current and z_future")`
- L30: `    z_current = data["z_current"].astype(np.float32)`
- L31: `    z_future = data["z_future"].astype(np.float32)`
- L36: `        action = np.zeros((z_current.shape[0], 0), dtype=np.float32)`
- L38: `    x = np.concatenate([z_current, action], axis=1).astype(np.float32)`
- L39: `    y = z_future.astype(np.float32)`
- L67: `def make_loader(x, y, batch_size: int, shuffle: bool):`
- L70: `    return DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=shuffle, drop_last=False)`
- L77: `    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)`
- L111: `    loader = make_loader(train_x, train_y, batch_size=batch_size, shuffle=True)`
- L197: `    loader = DataLoader(TensorDataset(tx, ty), batch_size=batch_size, shuffle=True, drop_last=False)`
- L266: `    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)`

## scripts/trainval_gain_router.py
- L42: `            # In the TartanAir feature files, x is [z_current, action].`
- L64: `    if {"z_current", "z_future", "action"}.issubset(keys):`
- L65: `        x = np.concatenate([d["z_current"], d["action"]], axis=1).astype(np.float32)`
- L66: `        return x, d["z_future"].astype(np.float32), d["action"].astype(np.float32)`

## src/spsm/cli/train.py
- L28: `    z_preds, z_futures, clean_futures = [], [], []`
- L32: `        z_current = batch["z_current"].to(device)`
- L34: `        z_future = batch["z_future"].to(device)`
- L35: `        clean_future = batch.get("clean_future", z_future).to(device)`
- L39: `        out = model(z_current, action)`
- L41: `        residual_observed = torch.mean((out["z_pred"] - z_future) ** 2, dim=-1)`
- L45: `        z_futures.append(z_future.detach().cpu())`
- L54: `        "z_future": torch.cat(z_futures),`
- L93: `    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)`
- L94: `    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)`
- L121: `    np.savez(`
- L179: `        sample["z_current"].to(device),`

## src/spsm/datasets/feature_npz.py
- L14: `      - z_current: [N, D]`
- L16: `      - z_future: [N, D]`
- L17: `      - clean_future: [N, D] optional; defaults to z_future`
- L30: `        self.z_current = torch.as_tensor(data["z_current"], dtype=torch.float32)`
- L32: `        self.z_future = torch.as_tensor(data["z_future"], dtype=torch.float32)`
- L33: `        clean = data["clean_future"] if "clean_future" in data else data["z_future"]`
- L35: `        n = self.z_current.shape[0]`
- L42: `        if self.z_current.ndim != 2 or self.z_future.ndim != 2:`
- L43: `            raise ValueError("z_current and z_future must be 2D arrays [N, D].")`
- L46: `        if self.z_current.shape != self.z_future.shape:`
- L47: `            raise ValueError("z_current and z_future must have the same shape.")`
- L49: `            raise ValueError("action must have the same number of rows as z_current.")`
- L52: `        return self.z_current.shape[0]`
- L56: `        return int(self.z_current.shape[1])`
- L64: `            "z_current": self.z_current[idx],`
- L66: `            "z_future": self.z_future[idx],`

## src/spsm/datasets/synthetic.py
- L37: `        z_future = A z_current + B action + noise`
- L53: `        self.z_current = torch.randn(config.n_samples, latent_dim, generator=g)`
- L68: `        clean_future = self.z_current @ self.A.T + self.action @ self.B + noise`
- L74: `        self.z_future = torch.where(observed_surprise[:, None], corrupted_future, clean_future)`
- L87: `            "z_current": self.z_current[idx],`
- L89: `            "z_future": self.z_future[idx],`

## src/spsm/datasets/tartanair.py
- L6: `class TartanAirLatentDataset(Dataset):`
- L10: `    placeholder for later frozen-feature trajectories.`
- L14: `        raise NotImplementedError("Public trajectory loader will be implemented after week-1 MRE.")`

## src/spsm/eval/latency.py
- L16: `    z_current: torch.Tensor,`
- L23: `        _ = model(z_current, action)`
- L24: `    if z_current.is_cuda:`
- L28: `        _ = model(z_current, action)`
- L29: `    if z_current.is_cuda:`
- L34: `        "batch_size": int(z_current.shape[0]),`
- L35: `        "samples_per_second": float(z_current.shape[0] * repeats / elapsed),`

## src/spsm/features/patch_encoder.py
- L15: `    pipeline before plugging in DINO/V-JEPA/TartanAir features.`

## src/spsm/features/torchvision_encoder.py
- L48: `            self.transform = transforms.Compose(`

## src/spsm/models/predictor.py
- L32: `    def forward(self, z_current: torch.Tensor, action: torch.Tensor \| None = None) -> torch.Tensor:`
- L34: `            x = torch.cat([z_current, action], dim=-1)`
- L36: `            x = z_current`
- L67: `    def forward(self, z_current: torch.Tensor, action: torch.Tensor \| None = None) -> dict[str, torch.Tensor]:`
- L68: `        z_pred = self.predictor(z_current, action)`
- L69: `        reliability_logit = self.reliability(z_current, z_pred, action)`

## src/spsm/models/reliability.py
- L28: `        # domain flags, etc.). We therefore expose simple action statistics in`
- L42: `        z_current: torch.Tensor,`
- L46: `        parts = [z_current, z_pred]`

## src/spsm/training/trainer.py
- L44: `                z_current = batch["z_current"].to(self.device)`
- L46: `                z_future = batch["z_future"].to(self.device)`
- L51: `                out = self.model(z_current, action)`
- L52: `                pred_loss = masked_mse(out["z_pred"], z_future, valid_mask)`
- L68: `                bs = z_current.shape[0]`

## src/spsm.egg-info/SOURCES.txt
- L20: `src/spsm/datasets/tartanair.py`

## tests/test_dataloaders.py
- L7: `    assert item["z_current"].shape == (12,)`
- L9: `    assert item["z_future"].shape == (12,)`

## tests/test_feature_npz_dataset.py
- L9: `    path = tmp_path / "features.npz"`
- L10: `    np.savez(`
- L12: `        z_current=np.zeros((4, 8), dtype=np.float32),`
- L14: `        z_future=np.ones((4, 8), dtype=np.float32),`
- L23: `    assert item["z_current"].shape[0] == 8`

## tests/test_reproducibility.py
- L10: `    assert torch.allclose(a.z_current, b.z_current)`
- L11: `    assert torch.allclose(a.z_future, b.z_future)`

# NPZ content audit

## outputs/tartanair_dinov2_features/features.npz
size_mb: 10.41
keys:
- z_current: shape=(13170, 384), dtype=float32
- z_future: shape=(13170, 384), dtype=float32
- action: shape=(13170, 7), dtype=float32
- expected_unreliable: shape=(13170,), dtype=float32
- observed_surprise: shape=(13170,), dtype=float32
- encoder: scalar='dinov2_vits14', dtype=<U13
- latent_dim: scalar=384, dtype=int64
- action_dim: scalar=7, dtype=int64
- source: scalar='tartanair_dinov2', dtype=<U16
id_like_keys: NONE

## outputs/tartanair_dinov2_features/features_train.npz
size_mb: 26.95
keys:
- z_current: shape=(9878, 384), dtype=float32
- z_future: shape=(9878, 384), dtype=float32
- action: shape=(9878, 7), dtype=float32
- expected_unreliable: shape=(9878,), dtype=float32
- observed_surprise: shape=(9878,), dtype=float32
- encoder: scalar='dinov2_vits14', dtype=<U13
- latent_dim: scalar=384, dtype=int64
- action_dim: scalar=7, dtype=int64
- source: scalar='tartanair_dinov2', dtype=<U16
id_like_keys: NONE

## outputs/tartanair_dinov2_features/features_val.npz
size_mb: 8.98
keys:
- z_current: shape=(3292, 384), dtype=float32
- z_future: shape=(3292, 384), dtype=float32
- action: shape=(3292, 7), dtype=float32
- expected_unreliable: shape=(3292,), dtype=float32
- observed_surprise: shape=(3292,), dtype=float32
- encoder: scalar='dinov2_vits14', dtype=<U13
- latent_dim: scalar=384, dtype=int64
- action_dim: scalar=7, dtype=int64
- source: scalar='tartanair_dinov2', dtype=<U16
id_like_keys: NONE

## outputs/tartanair_dinov2_features/features_train_errorrel.npz
size_mb: 26.99
keys:
- z_current: shape=(9878, 384), dtype=float32
- z_future: shape=(9878, 384), dtype=float32
- action: shape=(9878, 7), dtype=float32
- expected_unreliable: shape=(9878,), dtype=float32
- observed_surprise: shape=(9878,), dtype=float32
- encoder: scalar='dinov2_vits14', dtype=<U13
- latent_dim: scalar=384, dtype=int64
- action_dim: scalar=7, dtype=int64
- source: scalar='tartanair_dinov2', dtype=<U16
- cheap_prediction_error: shape=(9878,), dtype=float32
- error_reliability_threshold: scalar=0.8371084332466125, dtype=float32
- reliability_target_kind: scalar='cheap_predictor_error_quantile', dtype=<U30
id_like_keys: NONE

## outputs/tartanair_dinov2_features/features_val_errorrel.npz
size_mb: 8.99
keys:
- z_current: shape=(3292, 384), dtype=float32
- z_future: shape=(3292, 384), dtype=float32
- action: shape=(3292, 7), dtype=float32
- expected_unreliable: shape=(3292,), dtype=float32
- observed_surprise: shape=(3292,), dtype=float32
- encoder: scalar='dinov2_vits14', dtype=<U13
- latent_dim: scalar=384, dtype=int64
- action_dim: scalar=7, dtype=int64
- source: scalar='tartanair_dinov2', dtype=<U16
- cheap_prediction_error: shape=(3292,), dtype=float32
- error_reliability_threshold: scalar=0.8371084332466125, dtype=float32
- reliability_target_kind: scalar='cheap_predictor_error_quantile', dtype=<U30
id_like_keys: NONE

## outputs/tartanair_resnet18_features_full/features.npz
size_mb: 13.41
keys:
- z_current: shape=(13170, 512), dtype=float32
- z_future: shape=(13170, 512), dtype=float32
- action: shape=(13170, 7), dtype=float32
- expected_unreliable: shape=(13170,), dtype=float32
- observed_surprise: shape=(13170,), dtype=float32
- encoder: scalar='resnet18', dtype=<U8
- latent_dim: scalar=512, dtype=int64
- action_dim: scalar=7, dtype=int64
- source: scalar='tartanair_tiny', dtype=<U14
id_like_keys: NONE

## outputs/tartanair_resnet18_features_full/features_train.npz
size_mb: 34.83
keys:
- z_current: shape=(9878, 512), dtype=float32
- z_future: shape=(9878, 512), dtype=float32
- action: shape=(9878, 7), dtype=float32
- expected_unreliable: shape=(9878,), dtype=float32
- observed_surprise: shape=(9878,), dtype=float32
- encoder: scalar='resnet18', dtype=<U8
- latent_dim: scalar=512, dtype=int64
- action_dim: scalar=7, dtype=int64
- source: scalar='tartanair_tiny', dtype=<U14
id_like_keys: NONE

## outputs/tartanair_resnet18_features_full/features_val.npz
size_mb: 11.62
keys:
- z_current: shape=(3292, 512), dtype=float32
- z_future: shape=(3292, 512), dtype=float32
- action: shape=(3292, 7), dtype=float32
- expected_unreliable: shape=(3292,), dtype=float32
- observed_surprise: shape=(3292,), dtype=float32
- encoder: scalar='resnet18', dtype=<U8
- latent_dim: scalar=512, dtype=int64
- action_dim: scalar=7, dtype=int64
- source: scalar='tartanair_tiny', dtype=<U14
id_like_keys: NONE

# Bestval metrics provenance

## outputs/bestval_dinov2_cheap10_exp80_seed0/metrics.json
- seed: 0
- train: outputs/tartanair_dinov2_features/features_train.npz
- val: outputs/tartanair_dinov2_features/features_val.npz
- data: {'train': {'n_samples': 9878, 'input_dim': 391, 'latent_dim': 384, 'action_dim': 7}, 'val': {'n_samples': 3292, 'input_dim': 391, 'latent_dim': 384, 'action_dim': 7}}
- best_policy: threshold=0.50
- best_error: 1.2079479694366455
- best_compute: 1.7982989064398542
- best_selected: 0.2660996354799514
- best_utility: -1.2798799256942397

## outputs/bestval_dinov2_cheap10_exp80_seed1/metrics.json
- seed: 1
- train: outputs/tartanair_dinov2_features/features_train.npz
- val: outputs/tartanair_dinov2_features/features_val.npz
- data: {'train': {'n_samples': 9878, 'input_dim': 391, 'latent_dim': 384, 'action_dim': 7}, 'val': {'n_samples': 3292, 'input_dim': 391, 'latent_dim': 384, 'action_dim': 7}}
- best_policy: threshold=0.50
- best_error: 1.2086325883865356
- best_compute: 1.864823815309842
- best_selected: 0.2882746051032807
- best_utility: -1.2832255409989293

## outputs/bestval_dinov2_cheap10_exp80_seed2/metrics.json
- seed: 2
- train: outputs/tartanair_dinov2_features/features_train.npz
- val: outputs/tartanair_dinov2_features/features_val.npz
- data: {'train': {'n_samples': 9878, 'input_dim': 391, 'latent_dim': 384, 'action_dim': 7}, 'val': {'n_samples': 3292, 'input_dim': 391, 'latent_dim': 384, 'action_dim': 7}}
- best_policy: threshold=0.50
- best_error: 1.2013319730758667
- best_compute: 2.0534629404617255
- best_selected: 0.3511543134872418
- best_utility: -1.2834704906943357

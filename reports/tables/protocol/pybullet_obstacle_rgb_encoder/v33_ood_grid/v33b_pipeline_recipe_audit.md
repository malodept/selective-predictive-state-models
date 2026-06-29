# SPSM v33B pipeline recipe audit

This audit extracts the existing v5_3_ood generation recipe before expanding to a systematic OOD grid.

## Script argument definitions


### `scripts/create_pybullet_obstacle_intervention_rgb_dataset.py`

```
30:     parser = argparse.ArgumentParser()
31:     parser.add_argument("--out", type=Path, required=True)
32:     parser.add_argument("--report", type=Path, required=True)
33:     parser.add_argument("--groups", type=int, default=500)
34:     parser.add_argument("--image-size", type=int, default=96)
35:     parser.add_argument("--patch-grid", type=int, default=4)
36:     parser.add_argument("--horizon", type=int, default=36)
37:     parser.add_argument("--velocity", type=float, default=1.2)
38:     parser.add_argument("--num-blocked-directions", type=int, default=2)
39:     parser.add_argument("--seed", type=int, default=0)
```

### `scripts/extract_pybullet_rgb_dinov2_features.py`

```
16:     p = argparse.ArgumentParser()
17:     p.add_argument("--input", type=Path, required=True)
18:     p.add_argument("--output", type=Path, required=True)
19:     p.add_argument("--report", type=Path, required=True)
20:     p.add_argument("--model", default="dinov2_vits14")
21:     p.add_argument("--image-size", type=int, default=224)
22:     p.add_argument("--batch-size", type=int, default=64)
23:     p.add_argument("--device", default="cuda")
24:     p.add_argument("--keep-rgb", action="store_true")
```

### `scripts/pool_dinov2_patchtokens.py`

```
11:     p = argparse.ArgumentParser()
12:     p.add_argument("--input", type=Path, required=True)
13:     p.add_argument("--output", type=Path, required=True)
14:     p.add_argument("--report", type=Path, required=True)
15:     p.add_argument("--source-grid", type=int, default=16)
16:     p.add_argument("--target-grid", type=int, default=4)
```

### `scripts/mine_state_disjoint_mixed_hard_groups.py`

```
192:     parser = argparse.ArgumentParser()
193:     parser.add_argument("--data", type=Path, required=True)
194:     parser.add_argument("--out", type=Path, required=True)
195:     parser.add_argument("--report", type=Path, required=True)
196:     parser.add_argument("--train-groups", type=int, default=8000)
197:     parser.add_argument("--val-groups", type=int, default=1000)
198:     parser.add_argument("--test-groups", type=int, default=1000)
199:     parser.add_argument("--same-state-negatives", type=int, default=2)
200:     parser.add_argument("--same-action-negatives", type=int, default=2)
201:     parser.add_argument("--max-pos-dist", type=float, default=0.15)
202:     parser.add_argument("--prefer-block-mismatch", action="store_true")
203:     parser.add_argument("--seed", type=int, default=0)
```

### `scripts/eval_mixed_hard_tieaware_moving_only.py`

```
15:     p = argparse.ArgumentParser()
16:     p.add_argument("--data", type=Path, required=True)
17:     p.add_argument("--groups", type=Path, required=True)
18:     p.add_argument("--checkpoint", type=Path, required=True)
19:     p.add_argument("--out", type=Path, required=True)
20:     p.add_argument("--batch-groups", type=int, default=128)
21:     p.add_argument("--eps", type=float, default=1e-8)
22:     p.add_argument("--device", default="cuda")
```

### `scripts/eval_state_disjoint_confidence_diagnostics.py`

```
61:     p = argparse.ArgumentParser()
62:     p.add_argument("--data", type=Path, required=True)
63:     p.add_argument("--groups", type=Path, required=True)
64:     p.add_argument("--checkpoint", type=Path, required=True)
65:     p.add_argument("--out", type=Path, required=True)
66:     p.add_argument("--batch-groups", type=int, default=128)
67:     p.add_argument("--temperature", type=float, default=0.02)
68:     p.add_argument("--eps", type=float, default=1e-8)
69:     p.add_argument("--device", default="cuda")
```

## Existing v5_3_ood logs: first/last useful lines


### `logs/v5_3_ood/create_block2_h72_seed11.log`

```
- latent shape: `(16, 19)`
- current RGB shape: `(96, 96, 3)`
- future RGB shape: `(96, 96, 3)`
- horizon: `72`
- velocity: `1.2`
- blocked directions per state: `2`
- latent shape: `(16, 19)`
- current RGB shape: `(96, 96, 3)`
- future RGB shape: `(96, 96, 3)`
- horizon: `72`
- velocity: `1.2`
- blocked directions per state: `2`
```

### `logs/v5_3_ood/create_block2_v18_seed12.log`

```
- latent shape: `(16, 19)`
- current RGB shape: `(96, 96, 3)`
- future RGB shape: `(96, 96, 3)`
- horizon: `36`
- velocity: `1.8`
- blocked directions per state: `2`
- latent shape: `(16, 19)`
- current RGB shape: `(96, 96, 3)`
- future RGB shape: `(96, 96, 3)`
- horizon: `36`
- velocity: `1.8`
- blocked directions per state: `2`
```

### `logs/v5_3_ood/create_block3_h36_seed13.log`

```
- latent shape: `(16, 19)`
- current RGB shape: `(96, 96, 3)`
- future RGB shape: `(96, 96, 3)`
- horizon: `36`
- velocity: `1.2`
- blocked directions per state: `3`
- latent shape: `(16, 19)`
- current RGB shape: `(96, 96, 3)`
- future RGB shape: `(96, 96, 3)`
- horizon: `36`
- velocity: `1.2`
- blocked directions per state: `3`
```

### `logs/v5_3_ood/create_block3_h48_seed10.log`

```
- latent shape: `(16, 19)`
- current RGB shape: `(96, 96, 3)`
- future RGB shape: `(96, 96, 3)`
- horizon: `48`
- velocity: `1.2`
- blocked directions per state: `3`
- latent shape: `(16, 19)`
- current RGB shape: `(96, 96, 3)`
- future RGB shape: `(96, 96, 3)`
- horizon: `48`
- velocity: `1.2`
- blocked directions per state: `3`
```

### `logs/v5_3_ood/create_block3_h72_seed14.log`

```
- latent shape: `(16, 19)`
- current RGB shape: `(96, 96, 3)`
- future RGB shape: `(96, 96, 3)`
- horizon: `72`
- velocity: `1.2`
- blocked directions per state: `3`
- latent shape: `(16, 19)`
- current RGB shape: `(96, 96, 3)`
- future RGB shape: `(96, 96, 3)`
- horizon: `72`
- velocity: `1.2`
- blocked directions per state: `3`
```

### `logs/v5_3_ood/extract_dinov2_block2_h72_seed11.log`

```
input: outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_h72_seed11/pybullet_obstacle_rgb_v1_ood_block2_h72_seed11.npz
current_rgb: (10000, 96, 96, 3) uint8
future_rgb : (10000, 96, 96, 3) uint8
loading DINOv2: dinov2_vits14
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
extract current
extract 64/10000
extract 128/10000
extract 192/10000
extract 256/10000
extract 320/10000
extract 384/10000
extract 448/10000
extract 512/10000
extract 576/10000
extract 640/10000
extract 704/10000
extract 768/10000
extract 832/10000
extract 896/10000
extract 960/10000
extract 1024/10000
extract 1088/10000
extract 1152/10000
extract 1216/10000
extract 1280/10000
extract 1344/10000
extract 1408/10000
extract 1472/10000
extract 1536/10000
extract 1600/10000
extract 1664/10000
extract 1728/10000
extract 1792/10000
extract 1856/10000
...
extract 8832/10000
extract 8896/10000
extract 8960/10000
extract 9024/10000
extract 9088/10000
extract 9152/10000
extract 9216/10000
extract 9280/10000
extract 9344/10000
extract 9408/10000
extract 9472/10000
extract 9536/10000
extract 9600/10000
extract 9664/10000
extract 9728/10000
extract 9792/10000
extract 9856/10000
extract 9920/10000
extract 9984/10000
extract 10000/10000
reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_3_ood/block2_h72_seed11/dinov2_vits14.md
# PyBullet obstacle DINOv2 feature extraction

- input: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_h72_seed11/pybullet_obstacle_rgb_v1_ood_block2_h72_seed11.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_h72_seed11/pybullet_obstacle_rgb_v1_ood_block2_h72_seed11_dinov2_vits14.npz`
- model: `dinov2_vits14`
- image size: `224`
- z_current: `(10000, 256, 384)` `float16`
- z_future: `(10000, 256, 384)` `float16`
- cls_current: `(10000, 384)` `float16`
- cls_future: `(10000, 384)` `float16`
- action: `(10000, 2)`
- candidate groups: `(2000, 5)`
- output size GB: `3.6376`

## Interpretation

This replaces engineered PyBullet patch tokens with frozen DINOv2 patch-token representations extracted from RGB observations.
The resulting file is compatible with the exact-intervention oracle and ablation scripts.

```

### `logs/v5_3_ood/extract_dinov2_block2_v18_seed12.log`

```
input: outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_v18_seed12/pybullet_obstacle_rgb_v1_ood_block2_v18_seed12.npz
current_rgb: (10000, 96, 96, 3) uint8
future_rgb : (10000, 96, 96, 3) uint8
loading DINOv2: dinov2_vits14
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
extract current
extract 64/10000
extract 128/10000
extract 192/10000
extract 256/10000
extract 320/10000
extract 384/10000
extract 448/10000
extract 512/10000
extract 576/10000
extract 640/10000
extract 704/10000
extract 768/10000
extract 832/10000
extract 896/10000
extract 960/10000
extract 1024/10000
extract 1088/10000
extract 1152/10000
extract 1216/10000
extract 1280/10000
extract 1344/10000
extract 1408/10000
extract 1472/10000
extract 1536/10000
extract 1600/10000
extract 1664/10000
extract 1728/10000
extract 1792/10000
extract 1856/10000
...
extract 8832/10000
extract 8896/10000
extract 8960/10000
extract 9024/10000
extract 9088/10000
extract 9152/10000
extract 9216/10000
extract 9280/10000
extract 9344/10000
extract 9408/10000
extract 9472/10000
extract 9536/10000
extract 9600/10000
extract 9664/10000
extract 9728/10000
extract 9792/10000
extract 9856/10000
extract 9920/10000
extract 9984/10000
extract 10000/10000
reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_3_ood/block2_v18_seed12/dinov2_vits14.md
# PyBullet obstacle DINOv2 feature extraction

- input: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_v18_seed12/pybullet_obstacle_rgb_v1_ood_block2_v18_seed12.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_v18_seed12/pybullet_obstacle_rgb_v1_ood_block2_v18_seed12_dinov2_vits14.npz`
- model: `dinov2_vits14`
- image size: `224`
- z_current: `(10000, 256, 384)` `float16`
- z_future: `(10000, 256, 384)` `float16`
- cls_current: `(10000, 384)` `float16`
- cls_future: `(10000, 384)` `float16`
- action: `(10000, 2)`
- candidate groups: `(2000, 5)`
- output size GB: `3.6375`

## Interpretation

This replaces engineered PyBullet patch tokens with frozen DINOv2 patch-token representations extracted from RGB observations.
The resulting file is compatible with the exact-intervention oracle and ablation scripts.

```

### `logs/v5_3_ood/extract_dinov2_block3_h36_seed13.log`

```
input: outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h36_seed13/pybullet_obstacle_rgb_v1_ood_block3_h36_seed13.npz
current_rgb: (10000, 96, 96, 3) uint8
future_rgb : (10000, 96, 96, 3) uint8
loading DINOv2: dinov2_vits14
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
extract current
extract 64/10000
extract 128/10000
extract 192/10000
extract 256/10000
extract 320/10000
extract 384/10000
extract 448/10000
extract 512/10000
extract 576/10000
extract 640/10000
extract 704/10000
extract 768/10000
extract 832/10000
extract 896/10000
extract 960/10000
extract 1024/10000
extract 1088/10000
extract 1152/10000
extract 1216/10000
extract 1280/10000
extract 1344/10000
extract 1408/10000
extract 1472/10000
extract 1536/10000
extract 1600/10000
extract 1664/10000
extract 1728/10000
extract 1792/10000
extract 1856/10000
...
extract 8832/10000
extract 8896/10000
extract 8960/10000
extract 9024/10000
extract 9088/10000
extract 9152/10000
extract 9216/10000
extract 9280/10000
extract 9344/10000
extract 9408/10000
extract 9472/10000
extract 9536/10000
extract 9600/10000
extract 9664/10000
extract 9728/10000
extract 9792/10000
extract 9856/10000
extract 9920/10000
extract 9984/10000
extract 10000/10000
reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_3_ood/block3_h36_seed13/dinov2_vits14.md
# PyBullet obstacle DINOv2 feature extraction

- input: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h36_seed13/pybullet_obstacle_rgb_v1_ood_block3_h36_seed13.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h36_seed13/pybullet_obstacle_rgb_v1_ood_block3_h36_seed13_dinov2_vits14.npz`
- model: `dinov2_vits14`
- image size: `224`
- z_current: `(10000, 256, 384)` `float16`
- z_future: `(10000, 256, 384)` `float16`
- cls_current: `(10000, 384)` `float16`
- cls_future: `(10000, 384)` `float16`
- action: `(10000, 2)`
- candidate groups: `(2000, 5)`
- output size GB: `3.6375`

## Interpretation

This replaces engineered PyBullet patch tokens with frozen DINOv2 patch-token representations extracted from RGB observations.
The resulting file is compatible with the exact-intervention oracle and ablation scripts.

```

### `logs/v5_3_ood/extract_dinov2_block3_h48_seed10.log`

```
input: outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h48_seed10/pybullet_obstacle_rgb_v1_ood_block3_h48_seed10.npz
current_rgb: (10000, 96, 96, 3) uint8
future_rgb : (10000, 96, 96, 3) uint8
loading DINOv2: dinov2_vits14
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
extract current
extract 64/10000
extract 128/10000
extract 192/10000
extract 256/10000
extract 320/10000
extract 384/10000
extract 448/10000
extract 512/10000
extract 576/10000
extract 640/10000
extract 704/10000
extract 768/10000
extract 832/10000
extract 896/10000
extract 960/10000
extract 1024/10000
extract 1088/10000
extract 1152/10000
extract 1216/10000
extract 1280/10000
extract 1344/10000
extract 1408/10000
extract 1472/10000
extract 1536/10000
extract 1600/10000
extract 1664/10000
extract 1728/10000
extract 1792/10000
extract 1856/10000
...
extract 8832/10000
extract 8896/10000
extract 8960/10000
extract 9024/10000
extract 9088/10000
extract 9152/10000
extract 9216/10000
extract 9280/10000
extract 9344/10000
extract 9408/10000
extract 9472/10000
extract 9536/10000
extract 9600/10000
extract 9664/10000
extract 9728/10000
extract 9792/10000
extract 9856/10000
extract 9920/10000
extract 9984/10000
extract 10000/10000
reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_3_ood/block3_h48_seed10/dinov2_vits14.md
# PyBullet obstacle DINOv2 feature extraction

- input: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h48_seed10/pybullet_obstacle_rgb_v1_ood_block3_h48_seed10.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h48_seed10/pybullet_obstacle_rgb_v1_ood_block3_h48_seed10_dinov2_vits14.npz`
- model: `dinov2_vits14`
- image size: `224`
- z_current: `(10000, 256, 384)` `float16`
- z_future: `(10000, 256, 384)` `float16`
- cls_current: `(10000, 384)` `float16`
- cls_future: `(10000, 384)` `float16`
- action: `(10000, 2)`
- candidate groups: `(2000, 5)`
- output size GB: `3.6374`

## Interpretation

This replaces engineered PyBullet patch tokens with frozen DINOv2 patch-token representations extracted from RGB observations.
The resulting file is compatible with the exact-intervention oracle and ablation scripts.

```

### `logs/v5_3_ood/extract_dinov2_block3_h72_seed14.log`

```
input: outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h72_seed14/pybullet_obstacle_rgb_v1_ood_block3_h72_seed14.npz
current_rgb: (10000, 96, 96, 3) uint8
future_rgb : (10000, 96, 96, 3) uint8
loading DINOv2: dinov2_vits14
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/swiglu_ffn.py:51: UserWarning: xFormers is not available (SwiGLU)
  warnings.warn("xFormers is not available (SwiGLU)")
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/attention.py:33: UserWarning: xFormers is not available (Attention)
  warnings.warn("xFormers is not available (Attention)")
/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main/dinov2/layers/block.py:40: UserWarning: xFormers is not available (Block)
  warnings.warn("xFormers is not available (Block)")
extract current
extract 64/10000
extract 128/10000
extract 192/10000
extract 256/10000
extract 320/10000
extract 384/10000
extract 448/10000
extract 512/10000
extract 576/10000
extract 640/10000
extract 704/10000
extract 768/10000
extract 832/10000
extract 896/10000
extract 960/10000
extract 1024/10000
extract 1088/10000
extract 1152/10000
extract 1216/10000
extract 1280/10000
extract 1344/10000
extract 1408/10000
extract 1472/10000
extract 1536/10000
extract 1600/10000
extract 1664/10000
extract 1728/10000
extract 1792/10000
extract 1856/10000
...
extract 8832/10000
extract 8896/10000
extract 8960/10000
extract 9024/10000
extract 9088/10000
extract 9152/10000
extract 9216/10000
extract 9280/10000
extract 9344/10000
extract 9408/10000
extract 9472/10000
extract 9536/10000
extract 9600/10000
extract 9664/10000
extract 9728/10000
extract 9792/10000
extract 9856/10000
extract 9920/10000
extract 9984/10000
extract 10000/10000
reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_3_ood/block3_h72_seed14/dinov2_vits14.md
# PyBullet obstacle DINOv2 feature extraction

- input: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h72_seed14/pybullet_obstacle_rgb_v1_ood_block3_h72_seed14.npz`
- output: `outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h72_seed14/pybullet_obstacle_rgb_v1_ood_block3_h72_seed14_dinov2_vits14.npz`
- model: `dinov2_vits14`
- image size: `224`
- z_current: `(10000, 256, 384)` `float16`
- z_future: `(10000, 256, 384)` `float16`
- cls_current: `(10000, 384)` `float16`
- cls_future: `(10000, 384)` `float16`
- action: `(10000, 2)`
- candidate groups: `(2000, 5)`
- output size GB: `3.6375`

## Interpretation

This replaces engineered PyBullet patch tokens with frozen DINOv2 patch-token representations extracted from RGB observations.
The resulting file is compatible with the exact-intervention oracle and ablation scripts.

```

### `logs/v5_3_ood/pool_block2_h72_seed11.log`

```
- source shape: `(10000, 256, 384)`
- pooled shape: `(10000, 16, 384)`
- source shape: `(10000, 256, 384)`
- pooled shape: `(10000, 16, 384)`
```

### `logs/v5_3_ood/pool_block2_v18_seed12.log`

```
- source shape: `(10000, 256, 384)`
- pooled shape: `(10000, 16, 384)`
- source shape: `(10000, 256, 384)`
- pooled shape: `(10000, 16, 384)`
```

### `logs/v5_3_ood/pool_block3_h36_seed13.log`

```
- source shape: `(10000, 256, 384)`
- pooled shape: `(10000, 16, 384)`
- source shape: `(10000, 256, 384)`
- pooled shape: `(10000, 16, 384)`
```

### `logs/v5_3_ood/pool_block3_h48_seed10.log`

```
- source shape: `(10000, 256, 384)`
- pooled shape: `(10000, 16, 384)`
- source shape: `(10000, 256, 384)`
- pooled shape: `(10000, 16, 384)`
```

### `logs/v5_3_ood/pool_block3_h72_seed14.log`

```
- source shape: `(10000, 256, 384)`
- pooled shape: `(10000, 16, 384)`
- source shape: `(10000, 256, 384)`
- pooled shape: `(10000, 16, 384)`
```

### `logs/v5_3_ood/mine_block2_h72_seed11.log`

```
- prefer blocked mismatch: `True`
| --- | ---: |
| --- | ---: |
| blocked-mismatch fraction | 0.707313 |
- prefer blocked mismatch: `True`
| --- | ---: |
| --- | ---: |
| blocked-mismatch fraction | 0.707313 |
```

### `logs/v5_3_ood/mine_block2_v18_seed12.log`

```
- prefer blocked mismatch: `True`
| --- | ---: |
| --- | ---: |
| blocked-mismatch fraction | 0.704063 |
- prefer blocked mismatch: `True`
| --- | ---: |
| --- | ---: |
| blocked-mismatch fraction | 0.704063 |
```

### `logs/v5_3_ood/mine_block3_h36_seed13.log`

```
- prefer blocked mismatch: `True`
| --- | ---: |
| --- | ---: |
| blocked-mismatch fraction | 0.691688 |
- prefer blocked mismatch: `True`
| --- | ---: |
| --- | ---: |
| blocked-mismatch fraction | 0.691688 |
```

### `logs/v5_3_ood/mine_block3_h48_seed10.log`

```
- prefer blocked mismatch: `True`
| --- | ---: |
| --- | ---: |
| blocked-mismatch fraction | 0.690750 |
- prefer blocked mismatch: `True`
| --- | ---: |
| --- | ---: |
| blocked-mismatch fraction | 0.690750 |
```

### `logs/v5_3_ood/mine_block3_h72_seed14.log`

```
- prefer blocked mismatch: `True`
| --- | ---: |
| --- | ---: |
| blocked-mismatch fraction | 0.689875 |
- prefer blocked mismatch: `True`
| --- | ---: |
| --- | ---: |
| blocked-mismatch fraction | 0.689875 |
```

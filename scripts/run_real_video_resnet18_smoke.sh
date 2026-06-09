#!/usr/bin/env bash
set -euo pipefail

python scripts/extract_frames_from_videos.py \
  --input data/raw_videos \
  --output data/real_video_sequences \
  --fps 5 \
  --max-frames-per-video 120 \
  --image-size 224 \
  --overwrite

python scripts/extract_image_sequence_features.py \
  --input data/real_video_sequences \
  --output outputs/real_video_resnet18_features/features.npz \
  --encoder resnet18 \
  --resnet-weights imagenet \
  --max-gap 5 \
  --hard-gap 3 \
  --mismatch-prob 0.15

python scripts/split_feature_npz.py \
  --input outputs/real_video_resnet18_features/features.npz

python -m spsm.cli.train \
  --config configs/train/real_video_resnet18_smoke.yaml

python scripts/plot_synthetic_selective_compute.py \
  --run-dir outputs/real_video_resnet18_smoke \
  --figure-dir reports/figures/real_video_resnet18_smoke \
  --title-prefix "Real video ResNet18 smoke"

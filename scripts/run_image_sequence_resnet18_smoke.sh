#!/usr/bin/env bash
set -euo pipefail
python scripts/make_sample_image_sequences.py --output data/sample_image_sequences --n-sequences 12 --n-frames 16
python scripts/extract_image_sequence_features.py --input data/sample_image_sequences --output outputs/image_sequence_resnet18_features/features.npz --encoder resnet18 --resnet-weights imagenet --max-gap 3 --hard-gap 2
python scripts/split_feature_npz.py --input outputs/image_sequence_resnet18_features/features.npz
python -m spsm.cli.train --config configs/train/image_sequence_resnet18_smoke.yaml
python scripts/plot_synthetic_selective_compute.py --run-dir outputs/image_sequence_resnet18_smoke --figure-dir reports/figures/image_sequence_resnet18_smoke --title-prefix "ResNet18 image feature smoke"

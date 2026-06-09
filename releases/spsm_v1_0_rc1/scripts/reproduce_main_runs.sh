#!/usr/bin/env bash
set -euo pipefail

cd /shared/home/mdepastor/projects/spsm

CONTAINER="/shared/projects/phisat2/containers/phisat2.sif"

echo "Running tests..."
apptainer exec "$CONTAINER" python -m pytest tests

echo "Main configs:"
ls configs/train/tartanair_resnet18_full.yaml
ls configs/train/tartanair_resnet18_error_reliability.yaml
ls configs/train/tartanair_dinov2_full.yaml
ls configs/train/tartanair_dinov2_error_reliability.yaml

echo "Main outputs:"
ls outputs/tartanair_resnet18_full/metrics.json
ls outputs/tartanair_resnet18_error_reliability/metrics.json
ls outputs/tartanair_dinov2_full/metrics.json
ls outputs/tartanair_dinov2_error_reliability/metrics.json

echo "Regenerating teacher ablation table..."
apptainer exec "$CONTAINER" python scripts/teacher_ablation_main_table.py

echo "Done."

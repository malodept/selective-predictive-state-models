#!/usr/bin/env bash
set -euo pipefail
python -m spsm.cli.train --config configs/train/synthetic_selective_compute.yaml
python scripts/plot_synthetic_selective_compute.py

#!/usr/bin/env bash
#SBATCH --job-name=spsm_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
mkdir -p logs
CONFIG=${1:-configs/train/week1_mre.yaml}

# On the ESA environment, replace python with phipy if required by the project setup.
python -m spsm.cli.train --config "$CONFIG"

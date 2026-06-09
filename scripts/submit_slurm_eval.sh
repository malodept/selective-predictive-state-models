#!/usr/bin/env bash
#SBATCH --job-name=spsm_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
mkdir -p logs
echo "Standalone evaluation job is planned after week-1 MRE."

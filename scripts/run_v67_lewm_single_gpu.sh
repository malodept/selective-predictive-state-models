#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?mode required}"
SEED="${2:?seed required}"
N="${3:?n required}"
DIST="${4:-100}"

export PROJECT=/lustre/projects/1001
cd $PROJECT/code/spsm

source $PROJECT/envs/lewm310/bin/activate

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$PROJECT/code/external_worldmodels/le-wm:$PROJECT/code/external_worldmodels/stable-worldmodel:$PROJECT/code/spsm:${PYTHONPATH:-}
export STABLEWM_HOME=$PROJECT/stablewm_home
export HF_HOME=$PROJECT/cache/huggingface
export TORCH_HOME=$PROJECT/cache/torch
export SDL_VIDEODRIVER=dummy

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENCV_FOR_THREADS_NUM=1

python - <<'PY'
import torch
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

python scripts/audit_v67_official_lewm_pusht_history.py \
  --n "$N" \
  --seed "$SEED" \
  --device cuda \
  --candidate-mode "$MODE" \
  --dist-constraint "$DIST" \
  --out "reports/v67_external_lewm/lewm_${MODE}_d${DIST}_n${N}_seed${SEED}_gpu.md"

cat "reports/v67_external_lewm/lewm_${MODE}_d${DIST}_n${N}_seed${SEED}_gpu.md"

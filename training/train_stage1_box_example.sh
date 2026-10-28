#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/rl_games:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export WANDB_MODE="${WANDB_MODE:-offline}"

python stage1/train_stage1.py \
  --hand inspire_hand \
  --clip "${CLIP:-box-30-230-s01-use_01}" \
  -rn "${RETARGET:-para}" \
  -B 128 -am hybrid -imi 0.3 -imv 0.1 -bc 0.3 -ho 4 \
  --seed "${SEED:-42}" \
  -me "${EPOCHS:-1000}" -sf 1000 \
  --overlap_geom_penalty 20 \
  --overlap_geom_threshold 0.001 \
  --overlap_inside_backend grid \
  --overlap_grid_pitch 0.002 \
  -exp "${EXP_NAME:-fold_stage1_box_example}"

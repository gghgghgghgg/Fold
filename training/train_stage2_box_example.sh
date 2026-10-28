#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/rl_games:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export WANDB_MODE="${WANDB_MODE:-offline}"

BASE_CHECKPOINT="${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the selected Stage 1 checkpoint}"
COMMANDS="${COMMANDS:-$ROOT/force_command_estimator/data/box_gt_global_force_commands_full.npz}"
ENCODER_CHECKPOINT="${ENCODER_CHECKPOINT:-$ROOT/force_command_estimator/results/field_encoder/box_pointnet_300e/best_field_encoder.pt}"
test -s "$BASE_CHECKPOINT"
test -s "$COMMANDS"
test -s "$ENCODER_CHECKPOINT"

python stage2/train_stage2.py \
  --hand inspire_hand \
  --clip "${CLIP:-box-30-230-s01-use_01}" \
  -rn "${RETARGET:-para}" \
  -B 4096 -am hybrid -imi 0.3 -imv 0.1 -bc 0.3 -con 3.0 -ho 16 \
  --seed "${SEED:-42}" \
  -me "${EPOCHS:-10000}" -sf 1000 \
  --base_checkpoint "$BASE_CHECKPOINT" \
  --stage2_residual_scale 0.1 \
  --use_curriculum \
  --realtime_force_field_cmd \
  --field_command_data "$COMMANDS" \
  --field_encoder_checkpoint "$ENCODER_CHECKPOINT" \
  --field_active_top_k 60 \
  -exp "${EXP_NAME:-fold_stage2_box_example}"

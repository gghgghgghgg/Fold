#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

COMMANDS="${COMMANDS:-$ROOT/force_command_estimator/data/box_gt_global_force_commands_full.npz}"
OUT_DIR="${OUT_DIR:-$ROOT/force_command_estimator/results/field_encoder/box_pointnet_300e}"

test -s "$COMMANDS"
mkdir -p "$OUT_DIR"
python force_command_estimator/train_field_encoder.py \
  --data "$COMMANDS" \
  --epochs "${EPOCHS:-300}" \
  --seed "${SEED:-42}" \
  --device "${DEVICE:-cuda}" \
  --out-dir "$OUT_DIR"

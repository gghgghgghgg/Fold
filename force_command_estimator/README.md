# Force Commands and 128D Field Encoder

This directory contains the GT force-command generator and PointNet encoder used by Stage 2. Stage 2 solves the force field online from the current simulated object state and encodes it as a 128D observation. It does not use a precomputed latent sequence.

## Main files

- `generate_gt_commands.py` creates per-frame force commands from object states, surface anchors, and the object URDF.
- `field_encoder.py` defines the PointNet encoder.
- `train_field_encoder.py` trains the encoder.
- `../stage2/realtime_force_field.py` computes and encodes forces during Stage 2 training.

## Generate force commands

Prepare an anchor file and a `target_q.npy` containing object states. For the Box clip `[30,230)`:

```bash
python force_command_estimator/generate_gt_commands.py \
  --anchors global_anchor_force_field/data/box_global_anchors_200.npz \
  --urdf fold/assets/arctic/box/decomp/box_decomp.urdf \
  --target-q /path/to/box_target_q.npy \
  --start-frame 30 --num-frames 200 --active-top-k 60 \
  --topk-method normal_prefit \
  --out force_command_estimator/data/box_gt_global_force_commands_full.npz
```

Each command represents the transition from frame `i` to frame `i+1`. Set `--num-frames 0` to use all remaining transitions. Keep frame indices, object units, and URDF consistent.

## Train the encoder

```bash
python force_command_estimator/train_field_encoder.py \
  --data force_command_estimator/data/box_gt_global_force_commands_full.npz \
  --epochs 300 --seed 42 --device cuda \
  --out-dir force_command_estimator/results/field_encoder/box_pointnet_300e
```

The output `best_field_encoder.pt` is loaded by Stage 2.

## Stage 2 integration

Enable `--realtime_force_field_cmd` and provide `--field_command_data` and `--field_encoder_checkpoint`. Stage 2 solves forces on the global anchor set, selects the active top-k anchors, and encodes the result at each simulation step.

See the [reproduction guide](../docs/ENVIRONMENT_SETUP_AND_REPRODUCTION.md) for the complete workflow. Generated commands and encoder checkpoints are local training outputs and are not included in the source release.

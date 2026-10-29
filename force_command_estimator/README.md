# Force Commands and 128D Field Encoder

This directory contains the GT force-command generator and PointNet encoder used by Stage 2. Stage 2 solves the force field online from the current simulated object state and encodes it as a 128D observation. It does not use a precomputed latent sequence.

## Main files

- `generate_gt_commands.py` creates per-frame force commands from object states, surface anchors, and the object URDF.
- `derive_hand_distance_field_data.py` creates the hard 15 cm masked force labels used to train the matching encoder.
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

Derive the 15 cm masked force labels from these commands and the matching hand retarget:

```bash
python force_command_estimator/derive_hand_distance_field_data.py \
  --commands force_command_estimator/data/box_gt_global_force_commands_full.npz \
  --retarget fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt \
  --out force_command_estimator/data/box_hard15_force_fields.npz \
  --method hard15 --radius 0.15 --active-top-k 60
```

## Train the encoder

```bash
python force_command_estimator/train_field_encoder.py \
  --data force_command_estimator/data/box_hard15_force_fields.npz \
  --epochs 300 --seed 42 --device cuda \
  --out-dir force_command_estimator/results/field_encoder/box_hard15_pointnet_300e
```

The output `best_field_encoder.pt` is loaded by Stage 2.

## Stage 2 integration

The released Stage 2 example uses a 15 cm hand-proximity mask with up to 60 local anchors. Run it with `--field_hand_mask_radius 0.15`. If no anchor is within range, Stage 2 falls back to global top-k. Set the radius to `0` to use global top-k throughout.

New Stage 2 training defaults to 15 cm. Older saved checkpoints without a mask setting retain global-anchor evaluation behavior.

See the [reproduction guide](../docs/ENVIRONMENT_SETUP_AND_REPRODUCTION.md) for the complete workflow. Generated commands and encoder checkpoints are local training outputs and are not included in the source release.

# Training examples

This directory collects launch examples for the three learned components. The trainer implementations remain next to their model code (`stage1/`, `force_command_estimator/`, and `stage2/`); these scripts provide a single place to find the commands.

## Before training

1. Install the environment using the code at the top of the [repository README](../README.md).
2. Obtain data and models under their source terms. Install local files at the paths documented in the [full reproduction guide](../docs/ENVIRONMENT_SETUP_AND_REPRODUCTION.md). The local candidate workspace has Box, Ketchup, and Notebook examples staged under `fold/assets/`; their GitHub inclusion is pending written ARCTIC redistribution permission. See the [example-data manifest](../docs/EXAMPLE_DATA.md).
3. From the repository root, set `PYTHONPATH`, `SETUPTOOLS_USE_DISTUTILS`, and `WANDB_MODE` as shown in the README.
4. Prepare a complete local sequence, retarget trajectory, 200-anchor file, and force-command data. The full guide explains each preprocessing step and expected schema.

## Example order

The Box `[30,230)` examples use the same `para` retarget and seed 42. Run them from the repository root:

```bash
bash training/train_stage1_box_example.sh
```

After Stage 1 finishes, select its checkpoint and provide it to Stage 2. Generate force commands and train the encoder as described in the full guide, then run:

```bash
COMMANDS=force_command_estimator/data/box_gt_global_force_commands_full.npz \
  bash training/train_field_encoder_box_example.sh

BASE_CHECKPOINT=/path/to/selected_stage1_checkpoint.pth \
COMMANDS=force_command_estimator/data/box_gt_global_force_commands_full.npz \
ENCODER_CHECKPOINT=force_command_estimator/results/field_encoder/box_pointnet_300e/best_field_encoder.pt \
  bash training/train_stage2_box_example.sh
```

The examples use the reference batch sizes and epoch counts. They are templates, not a guarantee that those settings fit every GPU. The Stage 2 command consumes the exact selected Stage 1 checkpoint and the field encoder trained on the same clip/force-command dataset.

## Entry points

- [`train_stage1_box_example.sh`](train_stage1_box_example.sh): Stage 1 imitation policy.
- [`train_field_encoder_box_example.sh`](train_field_encoder_box_example.sh): 300-epoch 128D force-field encoder.
- [`train_stage2_box_example.sh`](train_stage2_box_example.sh): Stage 2 residual policy with curriculum and online force commands.

See [Environment Setup and Reproduction](../docs/ENVIRONMENT_SETUP_AND_REPRODUCTION.md) for the full data-to-evaluation workflow and configuration details.

# Training examples

These scripts run the Box `[30,230)` example with seed 42. Prepare the ARCTIC sequence and generate the force commands as described in the [reproduction guide](../docs/ENVIRONMENT_SETUP_AND_REPRODUCTION.md).

Run the stages in order from the repository root:

1. Stage 1 policy: `bash training/train_stage1_box_example.sh`
2. Train the 128D field encoder using the generated command file:

   ```bash
   COMMANDS=force_command_estimator/data/box_gt_global_force_commands_full.npz \
     bash training/train_field_encoder_box_example.sh
   ```

3. Stage 2 requires the selected Stage 1 checkpoint and trained encoder:

   ```bash
   BASE_CHECKPOINT=/path/to/stage1_checkpoint.pth \
   COMMANDS=force_command_estimator/data/box_gt_global_force_commands_full.npz \
   ENCODER_CHECKPOINT=force_command_estimator/results/field_encoder/box_pointnet_300e/best_field_encoder.pt \
     bash training/train_stage2_box_example.sh
   ```

The scripts use the reference training settings. Check the [full guide](../docs/ENVIRONMENT_SETUP_AND_REPRODUCTION.md) for installation, data preparation, evaluation, and configuration details.

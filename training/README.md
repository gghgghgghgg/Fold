# Training examples

These scripts run the Box `[30,230)` example with seed 42. Prepare the ARCTIC sequence and generate the force commands as described in the [reproduction guide](../docs/ENVIRONMENT_SETUP_AND_REPRODUCTION.md).

Run the stages in order from the repository root:

1. Stage 1 policy: `bash training/train_stage1_box_example.sh`
2. Derive the hard-15 cm force labels from the command file and retarget trajectory, then train the matching 128D encoder:

   ```bash
   python force_command_estimator/derive_hand_distance_field_data.py \
     --commands force_command_estimator/data/box_gt_global_force_commands_full.npz \
     --retarget fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt \
     --out force_command_estimator/data/box_hard15_force_fields.npz \
     --method hard15 --radius 0.15 --active-top-k 60
   COMMANDS=force_command_estimator/data/box_hard15_force_fields.npz \
     bash training/train_field_encoder_box_example.sh
   ```

3. Stage 2 requires the selected Stage 1 checkpoint and trained encoder:

   ```bash
   BASE_CHECKPOINT=/path/to/stage1_checkpoint.pth \
   COMMANDS=force_command_estimator/data/box_hard15_force_fields.npz \
   ENCODER_CHECKPOINT=force_command_estimator/results/field_encoder/box_hard15_pointnet_300e/best_field_encoder.pt \
     bash training/train_stage2_box_example.sh
   ```

The scripts use the reference training settings. Check the [full guide](../docs/ENVIRONMENT_SETUP_AND_REPRODUCTION.md) for installation, data preparation, evaluation, and configuration details.

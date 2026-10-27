Stage-1 Semantic Policy

This directory contains the FoLD Stage 1 imitation-policy training path.

Design goals:
- Keep the FoLD simulation environment and action interface consistent.
- Disable curriculum and object-side teacher assistance for stage 1.
- Use a ManipTrans-inspired semantic encoder over the current flat policy observation.
- Preserve output compatibility with the FoLD action space.

Files:
- obs_spec.py: reconstructs semantic observation slices from the flat policy observation.
- network_builder.py: per-field semantic encoders plus a shared PPO trunk.
- train_stage1.py: dedicated stage-1 entrypoint.

Example launch:

python stage1/train_stage1.py   --hand inspire_hand   --clip box-30-230-s01-u01   -B 4096   -am hybrid   -imi 0.3   -bc 0.3   --max_epochs 5000

Notes:
- train_stage1.py forces curriculum_cfg = {} and object_cfg.actuated = False.
- Output dimensions match the FoLD action interface.
- The saved run directory includes stage1_obs_spec.yaml for later stage-2 reuse.

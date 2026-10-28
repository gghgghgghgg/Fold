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

```bash
bash training/train_stage1_box_example.sh
```

This maintained example uses Box `[30,230)`, Inspire `para` retargeting, seed 42, batch 128, horizon 4, and 1000 epochs. Environment variables documented in [`training/README.md`](../training/README.md) can select the clip, retarget, seed, and epoch budget. Do not use the older B4096/5000 command from archived notes as the current example.

Notes:
- Stage 1 disables curriculum and object-side actuation.
- The saved run includes the observation specification used by Stage 2.

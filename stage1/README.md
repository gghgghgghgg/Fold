# Stage 1

Stage 1 trains the semantic imitation policy used as the base policy for Stage 2.

- `obs_spec.py`: semantic observation slices.
- `network_builder.py`: semantic encoders and PPO trunk.
- `train_stage1.py`: Stage 1 training entry point.

Run the Box example from the repository root:

```bash
bash training/train_stage1_box_example.sh
```

The launch settings and remaining training steps are documented in [`training/README.md`](../training/README.md).

# Global Anchor Force Field

This directory contains global surface-anchor generation and diagnostic tools. The current Stage 2 pipeline uses the realtime solver in `stage2/realtime_force_field.py`; this directory supplies the global anchor assets consumed by that path. The current method uses a global anchor set with active top-k selection. The hand-proximity mask remains an experimental variant and is not described as the released default.

## Generate anchors

Run from the repository root with authorized object templates available locally:

```bash
python global_anchor_force_field/generate_global_anchors.py \
  --meshes /path/to/arctic_data/data/meta/object_vtemplates/box/bottom.obj \
           /path/to/arctic_data/data/meta/object_vtemplates/box/top.obj \
  --part-names base lid --num-anchors 200 --scale 0.001 --seed 7 \
  --out global_anchor_force_field/data/box_global_anchors_200.npz
```

The `--scale` value must match the mesh units and object URDF coordinate system. Generate the cache on the machine that will use it; cached files can contain source mesh paths.

## Current training integration

The force-command generator consumes the anchor file and the object's authorized URDF. The Stage 2 trainer then solves the force field online from the current simulated object state, selects active anchors, and encodes the result for the policy. See [`force_command_estimator/README.md`](../force_command_estimator/README.md) and the [reproduction guide](../docs/ENVIRONMENT_SETUP_AND_REPRODUCTION.md) for the complete commands.

## Diagnostics

Older one-frame fitting, rollout, and visualization programs in this directory are standalone diagnostics. They are not required to run the current Stage 1 → force-command/encoder → Stage 2 pipeline. Their output and historical measurements should not be interpreted as current training results.

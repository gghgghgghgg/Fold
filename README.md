# FoLD: Force-field-guided Learning for Dexterous Manipulation

FoLD is a research pipeline for dexterous manipulation of articulated objects:

```text
Stage 1 imitation policy
  → object-surface anchor force optimization
  → PointNet force-field encoder (128D command)
  → Stage 2 residual policy with curriculum learning
  → deterministic full-clip evaluation
```

This source snapshot does not redistribute simulation hand/object models until their individual redistribution terms are cleared. It does not contain demonstrations, retargeted trajectories, generated force-command/anchor arrays, trained checkpoints, logs, or real-world robot-arm scene recordings. Obtain required external assets under their own terms; see the [English environment setup and reproduction guide](docs/ENVIRONMENT_SETUP_AND_REPRODUCTION.md) and [Chinese reproduction guide](docs/REPRODUCTION_CN.md). A clean clone is not runnable until required licensed assets and dataset files are installed locally.

## Layout

- `stage1/`: imitation-policy network and training entry points.
- `global_anchor_force_field/`: anchor generation, force optimization, rollout and visualization tools.
- `force_command_estimator/`: GT command generation and 128D PointNet encoder.
- `stage2/`: residual policy, curriculum integration and online force-field input.
- `fold/`: simulator environment, retargeting, evaluation and focused tests. Licensed robot/object assets are external inputs in this snapshot.
- `rl_games/`: runtime source needed by the training integration (upstream attribution/license retained).

## Environment and tests

Reference environment: Linux, Python 3.10, NVIDIA GPU, PyTorch 2.5.1 + CUDA 12.4, Genesis 0.3.3. Install the compatible CUDA PyTorch build first, then:

```bash
python -m pip install -r requirements-runtime.txt
python -m pip install -r requirements-dev.txt
export PYTHONPATH="$PWD/rl_games:$PWD"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export WANDB_MODE=offline
python -m pytest -q fold/tests
python -m compileall -q fold stage1 stage2 force_command_estimator global_anchor_force_field
```

The tests cover core observation, force-mask and temporal-retarget logic; they do not verify a complete GPU training run. Follow the reproduction guide and check the release-scope notes before redistributing third-party assets.

## Attribution

Parts of the simulation foundation derive from the upstream DexMachina project; retain its publication citation and license notices. FoLD is the name of this project and repository. Third-party hand models, object assets, and dependencies retain their respective licenses; their presence here is not a relicensing grant.

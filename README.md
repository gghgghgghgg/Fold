# FoLD: Force-field-guided Learning for Dexterous Manipulation

FoLD is a research pipeline for dexterous manipulation of articulated objects:

```text
Stage 1 imitation policy
  → object-surface anchor force optimization
  → PointNet force-field encoder (128D command)
  → Stage 2 residual policy with curriculum learning
  → deterministic full-clip evaluation
```

This source snapshot does not redistribute third-party hand/object models, ARCTIC demonstrations, retargeted trajectories, generated force-command/anchor arrays, trained checkpoints, logs, or real-world robot-arm scene recordings. Some upstream research repositories include these assets; obtain them from their original sources under the applicable terms. See [asset provenance and data instructions](docs/EXTERNAL_ASSETS.md), the [English environment setup and reproduction guide](docs/ENVIRONMENT_SETUP_AND_REPRODUCTION.md), and the [Chinese reproduction guide](docs/REPRODUCTION_CN.md).

Box, Ketchup, and Notebook sequence examples are staged locally under `fold/assets/` for private release preparation; they remain excluded from Git pending written ARCTIC redistribution permission. See the [example-data manifest](docs/EXAMPLE_DATA.md).

## Environment setup

The reference environment is Linux, Python 3.10, PyTorch 2.5.1 with CUDA 12.4, and Genesis 0.3.3. From the repository root:

```bash
conda create -n fold python=3.10 -y
conda activate fold
python -m pip install --upgrade pip

# Install the CUDA-enabled PyTorch wheel first.
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-runtime.txt
python -m pip install -r requirements-dev.txt

export PYTHONPATH="$PWD/rl_games:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export WANDB_MODE=offline
```

The [full environment and reproduction guide](docs/ENVIRONMENT_SETUP_AND_REPRODUCTION.md) covers licensed asset/data placement, preprocessing, force-command generation, Stage 1, encoder training, Stage 2, evaluation, and troubleshooting. Training launch examples are collected in [`training/`](training/README.md).

## Layout

- `stage1/`: imitation-policy network and Stage 1 trainer implementation.
- `global_anchor_force_field/`: anchor generation, force optimization, rollout and visualization tools.
- `force_command_estimator/`: GT command generation and 128D PointNet encoder.
- `stage2/`: residual policy, curriculum integration, online force-field input and Stage 2 trainer implementation.
- `training/`: collected example launch scripts for Stage 1, force-field encoder, and Stage 2.
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

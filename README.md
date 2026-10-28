# FoLD: Force-field-guided Learning for Dexterous Manipulation

FoLD is a pipeline for dexterous manipulation of articulated objects. This repository contains the training and evaluation code used in our work.

## Installation

We recommend Linux with Python 3.10. From the repository root:

```bash
conda create -n fold python=3.10 -y
conda activate fold
python -m pip install --upgrade pip

python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-runtime.txt
python -m pip install -r requirements-dev.txt

export PYTHONPATH="$PWD/rl_games:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export WANDB_MODE=offline
```

## Example

After installing the required data and models, run the Box Stage 1 example from the repository root:

```bash
bash training/train_stage1_box_example.sh
```

The [training examples](training/README.md) list the Stage 1, force-field encoder, and Stage 2 commands.

## Documentation

See the [environment setup and reproduction guide](docs/ENVIRONMENT_SETUP_AND_REPRODUCTION.md) for data placement, preprocessing, training, and evaluation. Asset sources and licensing notes are listed in [External Assets](docs/EXTERNAL_ASSETS.md).

## Attribution

The simulation foundation includes code derived from [DexMachina](https://github.com/MandiZhao/dexmachina); its license and attribution notices are retained. Third-party assets and dependencies remain subject to their original terms.

# FoLD Environment Setup and Reproduction

Install the environment, then prepare the ARCTIC sequence you want to use. The repository includes the Inspire hand model and currently tracks three processed examples with paired retarget trajectories. Follow [ARCTIC Data Preparation](EXTERNAL_ASSETS.md) to download and convert another sequence into the expected location. The ARCTIC data and its derived files remain subject to their source terms.

## 1. Install the environment

```bash
git clone https://github.com/gghgghgghgg/Fold.git FoLD
cd FoLD

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

Keep the retargeting package versions from `requirements-runtime.txt`: `dex-retargeting` 0.4.6 is pinned for this NumPy 1.26 environment. Version 0.5.0 requires NumPy 2 or newer.

Reference environment: Ubuntu Linux, Python 3.10, PyTorch 2.5.1 + CUDA 12.4, Genesis 0.3.3, and an NVIDIA driver compatible with CUDA 12.4. GPU smoke used a small batch and does not establish the memory requirement for the reference B=128/B=4096 runs. Start with a supported single GPU; reduce the environment batch only as an explicitly changed experiment setting and record it. Multi-GPU support is not established by this guide. Run the following checks from the repository root:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import genesis, igl, cv2; import rl_games; print("runtime imports OK")'
python -m pytest -q fold/tests
python -m compileall -q fold stage1 stage2 force_command_estimator global_anchor_force_field
```

A successful environment check prints CUDA availability as `True`, imports the runtime modules, and completes pytest with exit code 0. If the machine has no CUDA-capable GPU, Python package and CPU unit checks can still run, but GPU training is not validated.

## 2. ARCTIC sequence data

The repository includes the Inspire runtime hand model. Download and preprocess the ARCTIC sequence you want to use, then save its FoLD-converted file under:

```text
fold/assets/arctic/processed/<subject>/<object>_<action>.npy
```

For example: `fold/assets/arctic/processed/s01/box_use_01.npy`. Follow [ARCTIC Data Preparation](EXTERNAL_ASSETS.md). Existing examples are listed in [Example Data](EXAMPLE_DATA.md).

## 3. Prepare a sequence and retarget

For a new sequence, run `fold/retargeting/parallel_retarget.py` from the FoLD environment to create its full-sequence Inspire retarget file. For Box `s01/use_01`:

```bash
python fold/retargeting/parallel_retarget.py \
  --clip "box-0-889-s01-use_01" --hand inspire_hand \
  --retarget_type vector --save --save_name para
```

The output is `fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt`. Confirm that both processed data and retarget data cover the complete clip before generating force commands and training.

## 4. Generate anchors and force commands

Run from the FoLD repository root. Set `ARCTIC_ROOT` to the directory containing `data/meta/object_vtemplates/`:

```bash
export ARCTIC_ROOT=/path/to/arctic_data
export CLIP=box-30-230-s01-use_01
export RETARGET=para
export DEMO="$PWD/fold/assets/arctic/processed/s01/box_use_01.npy"
export OBJECT_URDF="$PWD/fold/assets/arctic/box/decomp/box_decomp.urdf"
export ANCHORS="$PWD/global_anchor_force_field/data/box_global_anchors_200.npz"
export TARGET_Q="$PWD/logs/local_data/box_target_q.npy"
export COMMANDS="$PWD/force_command_estimator/data/box_gt_global_force_commands_full.npz"
mkdir -p "$(dirname "$ANCHORS")" "$(dirname "$TARGET_Q")" "$(dirname "$COMMANDS")"
```

Extract object target states:

```bash
python - "$DEMO" "$TARGET_Q" <<'PY'
import sys
import numpy as np
p = np.load(sys.argv[1], allow_pickle=True).item()["params"]
q = np.concatenate((p["obj_trans"], p["obj_rot"], p["obj_arti"][:, None]), axis=1)
np.save(sys.argv[2], q)
print(q.shape)
PY
```

Generate 200 anchors and the 200-frame force-command file:

```bash
python global_anchor_force_field/generate_global_anchors.py \
  --meshes "$ARCTIC_ROOT/data/meta/object_vtemplates/box/bottom.obj" \
           "$ARCTIC_ROOT/data/meta/object_vtemplates/box/top.obj" \
  --part-names base lid --num-anchors 200 --scale 0.001 --seed 7 \
  --out "$ANCHORS"

python force_command_estimator/generate_gt_commands.py \
  --anchors "$ANCHORS" --urdf "$OBJECT_URDF" --target-q "$TARGET_Q" \
  --start-frame 30 --num-frames 200 --active-top-k 60 \
  --topk-method normal_prefit --out "$COMMANDS"
```

## 5. Train the example

The launch scripts are collected in [`training/`](../training/README.md). Run each step from the repository root.

Stage 1:

```bash
bash training/train_stage1_box_example.sh
```

Set `BASE_CHECKPOINT` to the selected Stage 1 checkpoint. Train the force-field encoder:

```bash
export ENCODER_DIR="$PWD/force_command_estimator/results/field_encoder/box_pointnet_300e"
COMMANDS="$COMMANDS" OUT_DIR="$ENCODER_DIR" \
  bash training/train_field_encoder_box_example.sh
```

Stage 2:

```bash
export BASE_CHECKPOINT=/path/to/stage1_checkpoint.pth
export ENCODER_CHECKPOINT="$ENCODER_DIR/best_field_encoder.pt"
BASE_CHECKPOINT="$BASE_CHECKPOINT" COMMANDS="$COMMANDS" \
ENCODER_CHECKPOINT="$ENCODER_CHECKPOINT" \
  bash training/train_stage2_box_example.sh
```

The example scripts use seed 42, Stage 1 batch size 128, Stage 2 batch size 4096, and the Box clip `[30,230)`. Change the script environment variables only when running a different configuration.

## 6. Evaluate a checkpoint

Set the paths to the selected Stage 2 checkpoint and the `params/env.pkl` saved with that run:

```bash
export STAGE2_CHECKPOINT=/path/to/stage2_checkpoint.pth
export STAGE2_ENV=/path/to/stage2_run/params/env.pkl
export EVAL_DIR="$PWD/reports/box200_eval"
```

Run the deterministic zero-gain evaluation:

```bash
python fold/rl/eval_rl_games.py \
  --hand inspire_hand --clip "$CLIP" --checkpoint "$STAGE2_CHECKPOINT" \
  --env_params_from "$STAGE2_ENV" --object_control_mode zero --seed 42 \
  --position_success_threshold 0.15 --rotation_success_threshold_deg 40 \
  --eval_task_name box-200 --eval_method ours --checkpoint_role best \
  --eval_data_dir "$EVAL_DIR"
```

Keep the matching clip, seed, environment file, and evaluation options for each method. The output directory contains the summary and per-frame evaluation data.


## Troubleshooting

- **CUDA unavailable:** run `nvidia-smi`, then check `torch.cuda.is_available()` and `torch.version.cuda`. Confirm the installed NVIDIA driver supports the CUDA wheel selected above. A successful CPU test does not validate GPU training.
- **`rl_games` import fails:** run from the repository root after exporting `PYTHONPATH="$PWD/rl_games:$PWD${PYTHONPATH:+:$PYTHONPATH}"`; do not install a different PyPI `rl-games` package over the vendored source.
- **Missing model/retarget file:** run the `test -s` checks in the relevant section and verify the exact subject, object, sequence, and retarget suffix. Generate FoLD processed data and full-sequence retarget output using their separate documented steps.
- **Numba cache is not writable:** point `NUMBA_CACHE_DIR` to a writable local directory, for example `export NUMBA_CACHE_DIR="$PWD/.cache/numba"`, then retry. Treat `NUMBA_DISABLE_JIT=1` as a diagnostic workaround only; it can change runtime performance.
- **Out of memory:** reduce `-B` and rerun as a changed configuration, recording the new value. Do not assume a small smoke batch predicts the memory needed by the reference configuration.

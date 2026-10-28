# FoLD Environment Setup and Reproduction

Follow the steps in order. Commands use the Box `s01/use_01` example. Install authorized data and models locally; this repository does not bundle ARCTIC sequences or third-party hand/object models. See [External Assets](EXTERNAL_ASSETS.md) for source links and paths.

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

Reference environment: Linux, Python 3.10, PyTorch 2.5.1 + CUDA 12.4, Genesis 0.3.3. Run the following checks from the repository root:

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -c 'import genesis, igl, cv2; print("imports OK")'
python -m pytest -q fold/tests
python -m compileall -q fold stage1 stage2 force_command_estimator global_anchor_force_field
```

## 2. Place local data and models

FoLD reads local inputs from `fold/assets/`. This directory is Git-ignored. Obtain files under their source terms and preserve the expected names and subdirectories.

```text
fold/assets/
├── arctic/
│   ├── box/                         # object URDF, collision/visual meshes
│   └── processed/s01/               # FoLD-processed sequence .npy files
├── inspire_hand/                    # hand URDF and meshes
└── retargeted/inspire_hand/s01/      # full-sequence retarget .pt files
```

For the example, prepare these paths:

```text
fold/assets/arctic/box/decomp/box_decomp.urdf
fold/assets/arctic/box/bottom_watertight_tiny.stl
fold/assets/arctic/box/top_watertight_tiny.stl
fold/assets/arctic/processed/s01/box_use_01.npy
fold/assets/inspire_hand/...
fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt
```

Read the ARCTIC and model-source terms before downloading. Do not add ARCTIC sequences, retarget files, or other restricted assets to Git without permission. The [asset guide](EXTERNAL_ASSETS.md) lists source URLs and local paths.

## 3. Process an ARCTIC sequence

Install and run ARCTIC preprocessing in its own ARCTIC environment. From the ARCTIC repository root:

```bash
export ARCTIC_REPO=/path/to/arctic
cd "$ARCTIC_REPO"
python scripts_data/process_seqs.py \
  --mano_p ./data/arctic_data/data/raw_seqs/s01/box_use_01.mano.npy \
  --export_verts
```

The expected intermediate file is:

```text
outputs/processed_verts/seqs/s01/box_use_01.npy
```

Run FoLD conversion from the FoLD repository root and its environment:

```bash
cd /path/to/FoLD
export ARCTIC_INTERMEDIATE="$ARCTIC_REPO/outputs/processed_verts/seqs/s01/box_use_01.npy"
python fold/retargeting/process_arctic.py \
  --processed_fname "$ARCTIC_INTERMEDIATE" --save \
  --save_dir fold/assets/arctic/processed
```

This creates `fold/assets/arctic/processed/s01/box_use_01.npy`.

Generate the full-sequence Inspire retarget file if it is not already available:

```bash
python fold/retargeting/parallel_retarget.py \
  --clip "box-0-889-s01-use_01" --hand inspire_hand \
  --retarget_type vector --save --save_name para
```

Expected output:

```text
fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt
```

Check inputs and frame coverage:

```bash
test -s fold/assets/arctic/box/decomp/box_decomp.urdf
test -s fold/assets/arctic/processed/s01/box_use_01.npy
test -s fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt

python - <<'PY'
import numpy as np
import torch
p = np.load("fold/assets/arctic/processed/s01/box_use_01.npy", allow_pickle=True).item()
r = torch.load("fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt",
               map_location="cpu", weights_only=False)
print("processed frames:", len(p["params"]["obj_trans"]))
print("retarget frames:", len(r["demo_data"]["obj_pos"]))
PY
```

Both files must cover the full clip used below.

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

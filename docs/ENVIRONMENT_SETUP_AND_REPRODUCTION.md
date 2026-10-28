# FoLD Environment Setup and Reproduction

Install the environment and prepare the required static assets first. For the included Box `s01/use_01` example, use the fast path in Section 3; for a new sequence or rebuilt trajectory, follow its preprocessing steps. 

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

## 2. Place local data and models

FoLD reads inputs from `fold/assets/`. The three processed sequences and three retarget trajectories in [Example Data](EXAMPLE_DATA.md) are tracked in the reviewed candidate tree. Other downloaded models and local files may be Git-ignored. Obtain and use every external file under its source terms; keep the expected names and subdirectories.

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
```

The runtime hand configuration loads `left_xyz_copy.urdf` and `right_xyz_copy.urdf`, whose package references require the listed STL and GLB trees. `retarget_config.yaml` is used only to regenerate retarget trajectories; it is not needed to load the supplied `.pt` trajectory for training. These URDFs are FoLD-adapted files: they add wrist joints/inertial data and use selected DexSuite GLB visual meshes. The exact upstream hand package revision used to create this local adaptation is not recorded, so a current upstream download is not guaranteed to be a drop-in replacement. Keep its source revision and file hashes with any locally prepared copy. Read all source terms before use; the [asset guide](EXTERNAL_ASSETS.md) lists sources and paths.

## 3. Process an ARCTIC sequence

### Fast path for the included Box example

Use the Box processed sequence and full-sequence `para` retarget provided in the repository. When reproducing with this example, skip the ARCTIC raw-data preprocessing, FoLD conversion, and retarget generation below. For a new sequence, add its processed file and generate the matching full-sequence retarget, then follow the preprocessing steps. In either case, you still need the Box simulator model, both Inspire hand models, and the ARCTIC Box object templates used to generate anchors. Run the asset check from the repository root:

```bash
python tools/check_reproduction_assets.py \
  --arctic-root /path/to/arctic_data
```

The check verifies the Box URDF mesh references, Inspire runtime URDF meshes, object templates, and that the processed sequence and retarget cover frames `[0,230)`. To recreate the retarget file, also install `retarget_config.yaml` and run with `--regenerate-retarget`; that configuration is not required when using the included retarget trajectory. For a new sequence or regenerated data, continue with the ARCTIC steps below.

Follow the [ARCTIC installation and data instructions](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md) and run its preprocessing in a separate `arctic` environment. Download the required raw sequence using the instructions for your ARCTIC checkout.

For the Box `s01/use_01` example, the raw file is `downloads/data/raw_seqs/s01/box_use_01.mano.npy`. From the ARCTIC repository root:

```bash
export ARCTIC_REPO=/path/to/arctic
cd "$ARCTIC_REPO"
conda activate arctic
python scripts_data/process_seqs.py \
  --mano_p downloads/data/raw_seqs/s01/box_use_01.mano.npy \
  --export_verts
```

The expected intermediate file is:

```text
outputs/processed_verts/seqs/s01/box_use_01.npy
```

To inspect the processed sequence with ARCTIC's viewer:

```bash
python scripts_data/visualizer.py \
  --no_image \
  --seq_p outputs/processed_verts/seqs/s01/box_use_01.npy \
  --mano --object
```

Convert the ARCTIC output to FoLD's expected format from the FoLD repository root and environment:

```bash
cd /path/to/FoLD
conda activate fold
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


## Troubleshooting

- **CUDA unavailable:** run `nvidia-smi`, then check `torch.cuda.is_available()` and `torch.version.cuda`. Confirm the installed NVIDIA driver supports the CUDA wheel selected above. A successful CPU test does not validate GPU training.
- **`rl_games` import fails:** run from the repository root after exporting `PYTHONPATH="$PWD/rl_games:$PWD${PYTHONPATH:+:$PYTHONPATH}"`; do not install a different PyPI `rl-games` package over the vendored source.
- **Missing model/retarget file:** run the `test -s` checks in the relevant section and verify the exact subject, object, sequence, and retarget suffix. Generate FoLD processed data and full-sequence retarget output using their separate documented steps.
- **Numba cache is not writable:** point `NUMBA_CACHE_DIR` to a writable local directory, for example `export NUMBA_CACHE_DIR="$PWD/.cache/numba"`, then retry. Treat `NUMBA_DISABLE_JIT=1` as a diagnostic workaround only; it can change runtime performance.
- **Out of memory:** reduce `-B` and rerun as a changed configuration, recording the new value. Do not assume a small smoke batch predicts the memory needed by the reference configuration.

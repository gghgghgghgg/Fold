# FoLD Environment Setup and Reproduction Guide

This guide describes how to install FoLD and reproduce the pipeline from local, authorized inputs: Stage 1 imitation, object-surface force-command generation, 128D force-field encoder training, Stage 2 residual learning with curriculum, and deterministic full-clip evaluation. The commands use Box-200 as a worked example; reproducing the full study requires repeating the pipeline with the exact hand, object, clip range, seed, and checkpoint-selection rule for each reported experiment.

The repository does not include third-party simulator hand/object models, ARCTIC sequences, processed or retargeted trajectories, generated anchor/force-command arrays, trained checkpoints, experiment logs, or real-world robot-arm recordings. A short Stage 1/Stage 2 smoke test has been run on a local candidate copy, but this is not a clean-machine reproduction of a full training run or a reproduction of the reported results. Obtain required assets separately under their source terms and install them at the documented paths.

## 1. Pipeline and artifacts

| Stage | Main input | Main output |
|---|---|---|
| Prepare data | Authorized ARCTIC sequence, object assets, retargeted hand trajectory | Full-length processed and retargeted sequences in the local asset tree |
| Stage 1 | Hand retarget trajectory and object demonstration | Imitation-policy checkpoint and saved environment configuration |
| Force-command generation | Object target states, object URDF, surface anchors | Per-frame 7D generalized force commands and anchor-force records |
| Field encoder | Force-command dataset | 128D PointNet-style encoder checkpoint and normalization metadata |
| Stage 2 | Stage 1 checkpoint, force commands, encoder, retarget trajectory | Residual policy checkpoint with curriculum state and training configuration |
| Evaluation | Selected checkpoint and matching saved environment configuration | Per-frame metrics and `metrics_summary.json` |

Stage 2 solves the force field online from the current simulator state and encodes it as a 128D policy input. The offline force-command data is used to train the encoder and provide command-related data; it is not a substitute for the online state-dependent force solve.

## 2. Reference software and hardware

The recorded reference environment is 64-bit Linux, Python 3.10, an NVIDIA GPU with a driver compatible with CUDA 12.4, PyTorch 2.5.1 with CUDA 12.4, Genesis 0.3.3, and the `rl_games` source vendored in this repository. The runtime dependency pins are in [`requirements-runtime.txt`](../requirements-runtime.txt); development tests are listed in [`requirements-dev.txt`](../requirements-dev.txt).

| Component | Reference version or source |
|---|---|
| Python | 3.10 |
| PyTorch | 2.5.1, CUDA 12.4 build |
| torchvision | 0.20.1 |
| Genesis | `genesis-world==0.3.3` |
| NumPy | 1.26.4 |
| Gym | 0.23.1 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| OSQP | 1.1.3 |
| Matplotlib | 3.10.8 |
| dex-retargeting | 0.5.0 |
| libigl | 2.5.1 |
| OpenCV | 4.11.0.86 |
| `rl_games` | Repository-vendored source; do not replace it with an unrelated PyPI version |

The required GPU memory depends on object, hand, batch size, and simulation settings. The reference commands below use `B=128` for Stage 1 and `B=4096` for Stage 2. First verify the installation with tests and a short smoke run. Reducing batch size can make a smoke test fit, but it changes the training configuration and does not reproduce the reference experiment.

## 3. Clone and install

Use a clean checkout of the FoLD repository. Create a separate Python environment instead of modifying a shared research environment:

```bash
git clone https://github.com/gghgghgghgg/Fold.git FoLD
cd FoLD

conda create -n fold python=3.10 -y
conda activate fold
python -m pip install --upgrade pip

# Install the CUDA-enabled PyTorch build first.
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

# Install the versions pinned for the FoLD runtime and tests.
python -m pip install -r requirements-runtime.txt
python -m pip install -r requirements-dev.txt
```

The PyTorch command follows the official PyTorch 2.5.1 CUDA 12.4 installation matrix. If the host driver or platform does not support this wheel, stop and resolve the driver/platform mismatch before substituting another CUDA or PyTorch version.

Example Stage 1, field-encoder, and Stage 2 launch scripts are collected in [`training/`](../training/README.md). They assume that the local files under `fold/assets/` and generated command/encoder inputs have already been prepared as described below.

From the repository root, configure the vendored RL package and offline experiment logging:

```bash
export PYTHONPATH="$PWD/rl_games:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export WANDB_MODE=offline
```

`WANDB_MODE=offline` is optional if online Weights & Biases logging is intentionally configured. Do not upload private data, trajectories, or checkpoints as run artifacts.

## 4. Verify the environment

Check the driver and device visibility:

```bash
nvidia-smi
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("PyTorch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

Check the pinned runtime packages and import the main simulator dependencies:

```bash
python - <<'PY'
from importlib.metadata import version
for package in ("genesis-world", "numpy", "gym", "scipy", "libigl", "opencv-python"):
    print(f"{package}=={version(package)}")
import genesis
import igl
import cv2
print("Genesis, libigl, and OpenCV imports succeeded")
PY

python -m pytest -q fold/tests
python -m compileall -q fold stage1 stage2 force_command_estimator global_anchor_force_field
```

These checks do not prove that Genesis can initialize the target GPU or that a full training run will complete. The first Genesis launch may compile GPU kernels and take longer than later launches. If Genesis/Numba reports a cache-locator error, first inspect the exact package versions and cache permissions. `NUMBA_DISABLE_JIT=1` has been used as a diagnostic workaround on one research server; it is not a default setting and its performance impact must be measured before using it for training.

## 5. Obtain and prepare local data and assets

Obtain ARCTIC data directly under the terms of the [official ARCTIC project](https://arctic.is.tue.mpg.de/) and its [data-processing instructions](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md). Read the [ARCTIC Data & Software License](https://github.com/zc-alexfan/arctic/blob/master/LICENSE) before downloading or using it. The license restricts use to specified non-commercial research, education, or artistic purposes and prohibits making the data and software available to third parties without prior written permission. Do not commit ARCTIC sequences or derived trajectories/commands to the FoLD repository. For source links and local asset paths for ARCTIC objects and robot hands, see the [external data and asset download guide](EXTERNAL_ASSETS.md).

### Convert an ARCTIC sequence to FoLD's processed format

ARCTIC provides the first preprocessing step: `scripts_data/process_seqs.py --export_verts` converts a raw MANO sequence to a vertex-export sequence. FoLD's [`process_arctic.py`](../fold/retargeting/process_arctic.py) then consumes that intermediate file, computes hand-object contact fields, converts object translations from millimeters to meters, adds quaternions, and writes the final FoLD dictionary. The [upstream ARCTIC data guide](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md) documents downloading and running its preprocessing script; use its setup instructions and license terms.

Run the ARCTIC command from the ARCTIC repository root, after installing its environment and obtaining the required ARCTIC, MANO, and object model files:

```bash
export ARCTIC_REPO=/path/to/arctic
cd "$ARCTIC_REPO"
python scripts_data/process_seqs.py \
  --mano_p ./data/arctic_data/data/raw_seqs/s01/box_use_01.mano.npy \
  --export_verts
```

The documented output is `outputs/processed_verts/seqs/s01/box_use_01.npy`. Then, from the FoLD repository root, run the FoLD contact/format conversion:

```bash
cd /path/to/FoLD
export ARCTIC_INTERMEDIATE="$ARCTIC_REPO/outputs/processed_verts/seqs/s01/box_use_01.npy"
python fold/retargeting/process_arctic.py \
  --processed_fname "$ARCTIC_INTERMEDIATE" --save \
  --save_dir fold/assets/arctic/processed
```

This writes `fold/assets/arctic/processed/s01/box_use_01.npy`, expected by the training pipeline. Verify its fields and frame count before retargeting or training. The two stages use separate environments: ARCTIC's environment for `process_seqs.py`, and FoLD's environment for `process_arctic.py`.

FoLD expects project-compatible processed data and a full-length retarget trajectory at these relative paths. The files must be prepared locally and are excluded from source control:

```text
fold/assets/arctic/processed/s01/box_use_01.npy
fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt
```

The processed `.npy` must be a dictionary compatible with the project loader. It includes at least `params` (`obj_trans`, `obj_rot`, `obj_arti`, and `obj_quat`) and `world_coord` fields used by the environment. Do not rename an upstream archive and treat it as this processed file. The retarget `.pt` must cover the complete source sequence, and its suffix (`para` or another supported retarget type) must match the `-rn` argument.

If compatible files already exist in an authorized local workspace, copy them into the expected paths:

```bash
mkdir -p fold/assets/arctic/processed/s01 \
  fold/assets/retargeted/inspire_hand/s01
cp "$ARCTIC_PROCESSED_SOURCE/s01/box_use_01.npy" \
  fold/assets/arctic/processed/s01/
cp "$RETARGET_SOURCE/s01/box_use_01_vector_para.pt" \
  fold/assets/retargeted/inspire_hand/s01/
```

If starting from upstream ARCTIC data, use the two-step conversion above. If the retarget result must be regenerated, use the repository retargeting pipeline with a compatible Genesis installation and the complete sequence. For the 889-frame Box example:

```bash
python fold/retargeting/parallel_retarget.py \
  --clip "box-0-889-s01-use_01" --hand inspire_hand \
  --retarget_type vector --save --save_name para
```

Before training, check the input files and sequence coverage:

```bash
test -s fold/assets/arctic/processed/s01/box_use_01.npy
test -s fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt
test -s fold/assets/arctic/box/decomp/box_decomp.urdf
test -s fold/assets/arctic/box/bottom_watertight_tiny.stl || \
  test -s fold/assets/arctic/box/bottom_watertight_tiny.obj
test -s fold/assets/arctic/box/top_watertight_tiny.stl || \
  test -s fold/assets/arctic/box/top_watertight_tiny.obj

python - <<'PY'
import numpy as np
import torch
p = np.load("fold/assets/arctic/processed/s01/box_use_01.npy", allow_pickle=True).item()
r = torch.load("fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt",
               map_location="cpu", weights_only=False)
print("Processed frames:", len(p["params"]["obj_trans"]))
print("Retarget frames:", len(r["demo_data"]["obj_pos"]))
PY
```

The Box-200 clip below uses frames `[30, 230)`, so both files must cover at least 230 frames. Keep the full sequence in the input files; the clip argument selects the training interval.

Simulation hand and object meshes, textures, and URDF files are also subject to their own licenses. Their presence in a source checkout does not grant redistribution rights. Real-world robot-arm recordings, camera/sensor recordings, and hardware calibration data are not required for this simulation pipeline and must not be added to the repository.

## 6. Generate anchors, object states, and force commands

Set paths for the local inputs and outputs. The ARCTIC object-template meshes must match the object URDF coordinate frame; verify their units before using `--scale 0.001`:

```bash
export ARCTIC_ROOT=/path/to/authorized/arctic_data
export DEMO=fold/assets/arctic/processed/s01/box_use_01.npy
export OBJECT_URDF="$PWD/fold/assets/arctic/box/decomp/box_decomp.urdf"
export ANCHORS="$PWD/global_anchor_force_field/data/box_global_anchors_200.npz"
export TARGET_Q="$PWD/logs/local_data/box_target_q.npy"
export COMMANDS="$PWD/force_command_estimator/data/box_gt_global_force_commands_full.npz"
mkdir -p "$(dirname "$TARGET_Q")" "$(dirname "$ANCHORS")" \
  "$(dirname "$COMMANDS")"
```

Create a full-sequence 7D object-state array `[x,y,z,rx,ry,rz,hinge]` from the processed parameters:

```bash
python - "$DEMO" "$TARGET_Q" <<'PY'
import sys
import numpy as np
data = np.load(sys.argv[1], allow_pickle=True).item()["params"]
q = np.concatenate((data["obj_trans"], data["obj_rot"],
                    data["obj_arti"][:, None]), axis=1)
np.save(sys.argv[2], q)
print("Saved target states:", q.shape)
PY
```

Generate the surface anchors locally. Do not copy an anchor `.npz` from another machine without checking its stored mesh paths; cached files may contain machine-specific absolute paths:

```bash
python global_anchor_force_field/generate_global_anchors.py \
  --meshes "$ARCTIC_ROOT/data/meta/object_vtemplates/box/bottom.obj" \
           "$ARCTIC_ROOT/data/meta/object_vtemplates/box/top.obj" \
  --part-names base lid --num-anchors 200 --scale 0.001 \
  --seed 7 --out "$ANCHORS"
```

Generate per-frame GT force commands for the Box-200 interval. Command row `i` represents the transition from target state `i` to `i+1`; confirm the source frame indexing before changing `--start-frame` or `--num-frames`:

```bash
python force_command_estimator/generate_gt_commands.py \
  --anchors "$ANCHORS" --urdf "$OBJECT_URDF" --target-q "$TARGET_Q" \
  --start-frame 30 --num-frames 200 --active-top-k 60 \
  --topk-method normal_prefit --out "$COMMANDS"
```

The force-command generator uses the object URDF's physical parameters and the configured wrench/solver settings. Preserve the generated input, configuration, command line, and output hash for every experiment.

## 7. Train Stage 1 and the force-field encoder

Set the clip and retarget type once so all stages use the same sequence interval:

```bash
export CLIP=box-30-230-s01-use_01
export RETARGET=para
```

Train the Stage 1 imitation policy:

```bash
python stage1/train_stage1.py \
  --hand inspire_hand --clip "$CLIP" -rn "$RETARGET" \
  -B 128 -am hybrid -imi 0.3 -imv 0.1 -bc 0.3 -ho 4 \
  --seed 42 \
  -me 1000 -sf 1000 --overlap_geom_penalty 20 \
  --overlap_geom_threshold 0.001 --overlap_inside_backend grid \
  --overlap_grid_pitch 0.002 -exp fold_stage1_box200
```

Select the Stage 1 checkpoint according to the experiment protocol, not just the latest filename. Record its epoch and SHA256, and retain the associated `params/agent.yaml`, `params/env.pkl`, and observation specification:

```bash
export BASE_CHECKPOINT=/absolute/path/to/stage1_checkpoint.pth
test -s "$BASE_CHECKPOINT"
sha256sum "$BASE_CHECKPOINT"
```

Train the 128D force-field encoder from the generated commands:

```bash
export ENCODER_DIR="$PWD/force_command_estimator/results/field_encoder/box_pointnet_300e"
python force_command_estimator/train_field_encoder.py \
  --data "$COMMANDS" --epochs 300 --seed 42 --device cuda \
  --out-dir "$ENCODER_DIR"
test -s "$ENCODER_DIR/best_field_encoder.pt"
```

Keep the encoder checkpoint and its normalization metadata together. The Stage 2 run must point to the encoder trained for the same command-data format and object setup.

## 8. Train Stage 2 with residual learning and curriculum

The following command is the Box-200 reference configuration:

```bash
python stage2/train_stage2.py \
  --hand inspire_hand --clip "$CLIP" -rn "$RETARGET" \
  -B 4096 -am hybrid -imi 0.3 -imv 0.1 -bc 0.3 -con 3.0 -ho 16 \
  --seed 42 \
  -me 10000 -sf 1000 --base_checkpoint "$BASE_CHECKPOINT" \
  --stage2_residual_scale 0.1 --use_curriculum \
  --realtime_force_field_cmd --field_command_data "$COMMANDS" \
  --field_encoder_checkpoint "$ENCODER_DIR/best_field_encoder.pt" \
  --field_active_top_k 60 -exp fold_stage2_box200
```

The released reference pipeline uses global anchor selection. A hand-proximity/local-anchor mask is an ongoing experiment and is intentionally not included in this paper-matched release snapshot.

For a curriculum resume, an RL-Games `.pth` file alone is insufficient. Save and restore the current curriculum bounds and the complete gain history along with the policy checkpoint, configuration, and optimizer state supported by the run. A restart that omits this curriculum state is a new or altered experiment, not an exact continuation.

## 9. Run deterministic full-clip evaluation

Evaluate the selected Stage 2 checkpoint using the matching saved environment parameters. Keep the policy checkpoint, clip, physical parameters, initialization, seed, control mode, and termination rules identical across methods. For the requested zero-gain evaluation protocol, use `--object_control_mode zero` and seed 42:

```bash
export STAGE2_CHECKPOINT=/absolute/path/to/selected_stage2_checkpoint.pth
export STAGE2_ENV=/absolute/path/to/stage2_run/params/env.pkl
export EVAL_DIR="$PWD/reports/box200_eval"
test -s "$STAGE2_CHECKPOINT"
test -s "$STAGE2_ENV"

python fold/rl/eval_rl_games.py \
  --hand inspire_hand --clip "$CLIP" --checkpoint "$STAGE2_CHECKPOINT" \
  --env_params_from "$STAGE2_ENV" --object_control_mode zero --seed 42 \
  --position_success_threshold 0.15 --rotation_success_threshold_deg 40 \
  --eval_task_name box-200 --eval_method ours --checkpoint_role best \
  --eval_data_dir "$EVAL_DIR"
```

Use the same full clip and saved environment for the baseline. Set `--eval_method baseline` and provide that method's selected checkpoint; do not compare a best checkpoint for one method against a final checkpoint for another unless the protocol explicitly calls for that comparison. Save the exact command, environment/configuration, checkpoint filename/epoch/SHA256, per-frame NPY/NPZ outputs, exit reason, and `metrics_summary.json`. Check `python fold/rl/eval_rl_games.py --help` for options in the checked-out revision.

## 10. Reproduce the complete experiment set

Box-200 is only the walkthrough example. To reproduce the full study, create an experiment manifest with one row per reported run and record at least:

- object, hand, subject/use, exact half-open frame interval, and retarget method;
- Stage 1, encoder, and Stage 2 seeds and training configurations;
- object URDF/mesh provenance and hashes, anchor-generation settings, and force-command settings;
- checkpoint-selection rule and checkpoint SHA256 for each method;
- evaluation seed, object-control mode, initialization, physical parameters, termination rule, and metric thresholds.

Run every method with matched inputs and evaluation settings. Do not infer a published result from a smoke test or from a single object. For each task, archive the experiment manifest, complete commands, logs, checkpoints, curriculum state, metrics, and generated command/encoder artifacts outside the source repository, subject to the relevant data and asset licenses.

## 11. Reproducibility limits and troubleshooting

Exact bitwise equality across different GPUs, driver versions, Genesis kernels, or asset revisions is not guaranteed. A successful import or one-epoch smoke test establishes that an entry point can execute; it does not establish convergence, policy quality, or reproduction of the reported metrics. Full result reproduction requires the same inputs, preprocessing, physical parameters, experiment settings, checkpoint choice, and evaluation protocol.

Common checks:

| Symptom | Check |
|---|---|
| `torch.cuda.is_available()` is false | Check `nvidia-smi`, the host driver, and that the installed PyTorch wheel is the CUDA 12.4 build. |
| Genesis fails during first initialization | Preserve the full traceback; check driver/runtime compatibility, package pins, GPU visibility, and writable kernel/cache directories. |
| GT command generation cannot load an OBJ | Regenerate anchors from local OBJ files and inspect `mesh_paths` stored in the `.npz`. |
| A sequence or clip is too short | Verify the full processed and retarget lengths and the clip's half-open frame interval. |
| Stage 2 cannot load a force field | Check command-data, encoder checkpoint, normalization metadata, object identity, and anchor dimensions. |
| A curriculum continuation diverges from the original run | Confirm that current bounds and the complete gain history were restored, not only the `.pth` checkpoint. |

The current repository smoke-test evidence is limited to short Stage 1 and Stage 2 runs. A fresh-clone, full-length multi-task reproduction must be validated separately on a machine with authorized data/assets and a compatible GPU environment.

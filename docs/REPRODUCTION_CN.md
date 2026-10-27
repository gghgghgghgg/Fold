# FoLD 环境与复现指南

本文给出一条可审计的参考流程：Stage 1 → GT 力场命令与 128D 编码器 → Stage 2 residual + curriculum → 确定性完整 clip eval。当前候选 Git 快照暂不分发仿真手部/物体模型，须待逐项确认授权后再纳入；ARCTIC演示数据、训练权重以及真机场景里的机械臂轨迹、相机/传感器采集等数据不由本仓库分发。本指南不把缺失数据伪装成仓库自带内容。

## 1. 参考环境

已记录的研究服务器配置：Ubuntu Linux、Python 3.10、NVIDIA GPU、PyTorch 2.5.1 + CUDA 12.4、Genesis 0.3.3、仓库内 vendored rl-games 1.6.1。驱动版本须兼容 CUDA 12.4；建议使用单卡先跑 smoke test。消费级 GPU 的显存需求随 `-B`（并行环境数）变化，正式设置 `Stage 1 B=128`、`Stage 2 B=4096` 可能需要多卡或降低 batch 并重新验证结果。

在 Linux 克隆仓库并建立 Python 3.10 环境：

```bash
git clone https://github.com/gghgghgghgg/Fold.git FoLD
cd FoLD
conda create -n fold python=3.10 -y
conda activate fold

# PyTorch 2.5.1 CUDA 12.4 wheel。先装它，再装其余依赖。
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-runtime.txt
python -m pip install -r requirements-dev.txt

export PYTHONPATH="$PWD/rl_games:$PWD"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export WANDB_MODE=offline
```

`rl_games` 运行源码位于仓库的 `rl_games/` 子目录；使用上面的 `PYTHONPATH` 让项目脚本直接从该源码导入，不需另外从 PyPI 安装。候选副本不含上游 Git 历史、文档图片、notebook 或测试目录。不同机器首次运行 Genesis 会编译/缓存 GPU kernel；应先完成环境检查和短 smoke test。若运行时出现 Genesis/Numba/Triton 导入或编译错误，请记录完整包版本、驱动、GPU 与 traceback；不要通过随意升级依赖来宣称复现成功。

核对安装版本：

```bash
python -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda, "available", torch.cuda.is_available())'
python -c 'import importlib.metadata as m; print("Genesis", m.version("genesis-world")); print("NumPy", m.version("numpy"))'
nvidia-smi
```

当前研究服务器在常规 import 下遇到过 `cannot cache function ... no locator available` 的 Genesis/Numba 导入问题；设置 `NUMBA_DISABLE_JIT=1` 后 import 验证通过。该开关可能影响性能，因此不作为默认配置；若在新环境复现该异常，先确认依赖版本和 Numba 缓存目录，再把它作为诊断 workaround，并实际验证训练速度/结果后再决定是否使用。

## 2. 外部数据与资产清单

训练前先依据数据集/资产自身许可取得文件。仓库不会分发训练演示、ARCTIC 原始数据或 checkpoint。请从[ARCTIC 官方项目页](https://arctic.is.tue.mpg.de/)申请数据，并参考上游[下载/解包说明](https://github.com/zc-alexfan/arctic/blob/master/docs/data/README.md)。ARCTIC 官方 Data & Software License 对数据及派生文件的第三方提供有限制；使用前阅读[数据许可](https://github.com/zc-alexfan/arctic/blob/master/LICENSE)，不要把序列、retarget 轨迹或从其生成的训练数据提交到 FoLD 仓库。

### 2.1 本地数据目录与准备顺序（Box-200 示例）

FoLD 目前按固定相对路径读取序列。克隆后在仓库根目录创建以下本地目录；把自己有权使用的项目兼容 processed sequence 和 retarget 轨迹放到这里。以下内容均被 `.gitignore` 排除，只供当前机器训练，不会随源码提交：

```text
fold/assets/arctic/processed/s01/box_use_01.npy
fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt

# 可选的其他本地验证序列
fold/assets/arctic/processed/s01/ketchup_use_01.npy
fold/assets/arctic/processed/s01/notebook_use_01.npy
fold/assets/retargeted/inspire_hand/s01/ketchup_use_01_vector_temporal_v1.pt
fold/assets/retargeted/inspire_hand/s01/notebook_use_01_vector_temporal_v1.pt
```

其中 processed `.npy` 必须是本项目可读的字典格式，至少含 `params`（`obj_trans`、`obj_rot`、`obj_arti`、`obj_quat`）和 `world_coord`（双手接触/关键点字段）；不要把官方原始压缩包直接改名当作此文件。若从 ARCTIC 上游原始文件开始，先按其官方数据处理说明生成项目使用的 processed 数据。`fold/retargeting/process_arctic.py` 只适用于已包含 `verts.left/right/object` 与 `parts_ids` 的中间数据，用来补充接触字段；它不是原始数据解包器，也不能单独生成 FoLD 所需的全部 `params/world_coord`。retarget `.pt` 应覆盖序列完整帧，名称中的 `para`/`temporal_v1` 必须与训练命令 `-rn` 一致；不能把某个截取片段的轨迹冒充完整序列。

示例：将本机已准备好的文件放入约定目录（请按自己的下载/解压路径修改源路径）：

```bash
mkdir -p fold/assets/arctic/processed/s01 fold/assets/retargeted/inspire_hand/s01
cp "$ARCTIC_PROCESSED_SOURCE/s01/box_use_01.npy" fold/assets/arctic/processed/s01/
cp "$RETARGET_SOURCE/s01/box_use_01_vector_para.pt" fold/assets/retargeted/inspire_hand/s01/

# 其他序列按需放入；检查文件名、长度和本地忽略规则
test -s fold/assets/arctic/processed/s01/box_use_01.npy
test -s fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt

# 检查 frame 数、必要字段及 retarget 完整序列长度。
python -c 'import numpy as np,torch; p="fold/assets/arctic/processed/s01/box_use_01.npy"; d=np.load(p,allow_pickle=True).item(); print("processed frames",len(d["params"]["obj_trans"]),"keys",list(d)); r=torch.load("fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt",map_location="cpu"); print("retarget demo frames",len(r["demo_data"]["obj_pos"]))'
```

若已有与当前复现实验一致的完整 retarget 结果，直接放入上表目标位置即可。若需在本机重新生成 `para`，先按完整序列帧数设置 `END_FRAME`（Box 示例为 889），然后从仓库根目录运行；Genesis retarget 需要可用的运行环境：

```bash
export END_FRAME=889
python fold/retargeting/parallel_retarget.py \
  --clip "box-0-${END_FRAME}-s01-use_01" --hand inspire_hand \
  --retarget_type vector --save --save_name para
```

输出应为 `fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt`。如果复现实验采用 `temporal_v1`，应使用该实验对应的 temporal retarget 处理配置生成同名版本；不要只把 `para` 文件改名为 `temporal_v1`。

训练脚本只会用 `--clip box-30-230-s01-use_01` 选择 30–229 帧；源 `.npy` 和 retarget 文件仍应保留完整序列。先检查 processed 序列帧数及 retarget 内 `demo_data` 长度覆盖 230 帧。对象 URDF/碰撞 mesh 应使用仓库内 `fold/assets/arctic/<object>/` 对应文件。

运行前检查全部输入是否已就位：

```bash
test -s fold/assets/arctic/processed/s01/box_use_01.npy
test -s fold/assets/retargeted/inspire_hand/s01/box_use_01_vector_para.pt
test -s fold/assets/arctic/box/box.urdf
```

后续由本机生成的文件放在以下目录：

| 生成物 | 本地路径 | 生成方式 |
|---|---|---|
| Box base/lid surface anchors | `global_anchor_force_field/data/box_global_anchors_200.npz` | 从已获准使用的 ARCTIC object template `bottom.obj`、`top.obj`运行 anchor generator；按 mesh 原始单位设置 scale |
| 物体 7D GT 状态 | `logs/local_data/box_target_q.npy` | 从 processed 文件 `params.obj_trans + params.obj_rot + params.obj_arti`按帧拼成 `[x,y,z,rx,ry,rz,hinge]`；旋转为 axis-angle |
| GT 力场命令 | `force_command_estimator/data/box_gt_global_force_commands_full.npz` | `generate_gt_commands.py`由 anchors、target-q、URDF 计算 |
| encoder 权重 | `force_command_estimator/results/field_encoder/box_pointnet_300e/best_field_encoder.pt` | `train_field_encoder.py`从 GT 命令数据训练 |
| Stage 1/2 checkpoint 与日志 | `logs/rl_games/inspire_hand/` | 对应 PPO 训练入口自动生成 |

**anchor 文件必须在当前机器本地重新生成。**缓存 `.npz` 会记录生成时 OBJ 的路径；从别的机器复制过来的缓存若包含 Windows/旧机器绝对路径，命令生成会因 mesh 无法读取而失败。生成后检查 `mesh_paths` 均能在当前系统访问，再运行 GT command 步骤。不要仅因为旧 `.npz` 存在就跳过这一步。

ARCTIC processed/retarget、anchors、force commands 和训练输出路径已在候选 `.gitignore` 中排除。不要用 `git add -f` 强行加入这些文件；公开源码仓库应只提交准备流程和代码，不提交 ARCTIC 数据或其派生训练数据。

至少需要：

| 输入 | 用途 | 常见位置/格式 |
|---|---|---|
| ARCTIC demonstration | 每帧 MANO/object 目标状态 | `fold/assets/arctic/processed/<subject>/<object>_<task>.npy`（由使用者本地准备） |
| 对象 URDF、视觉/碰撞 mesh、材质纹理 | Genesis 仿真、锚点与渲染 | 从经授权来源单独获取并放置到 `fold/assets/arctic/<object>/`；当前候选 Git 快照未包含 |
| Inspire 仿真手部 URDF/mesh/config | 双手仿真 | 从经授权来源单独获取并放置到 `fold/assets/inspire_hand/`；当前候选 Git 快照未包含 |
| 真机场景机械臂轨迹、相机/传感器记录 | 不属于仿真策略复现必需输入 | 不纳入 GitHub |
| retarget trajectory | Stage 1/2 手部目标轨迹 | `fold/assets/retargeted/inspire_hand/<subject>/<object>_use_01_vector_<retarget>.pt` |
| surface anchor 文件 | GT 命令求解及 realtime 场 | `global_anchor_force_field/data/<object>_global_anchors_*.npz`（在本机生成） |
| GT force-command 数据 | 训练 PointNet encoder / Stage 2 | `force_command_estimator/data/<object>_gt_global_force_commands_full.npz` |
| field encoder 权重 | Stage 2 在线生成 128D 命令 | `force_command_estimator/results/field_encoder/<run>/best_field_encoder.pt` |

仓库 `.gitignore` 会忽略大量几何、轨迹和数值数组扩展名。忽略规则不等于资产已经获得授权；也不保证 Git 历史中没有此前跟踪的文件。发布者需审查 `git ls-files` 和完整历史。建议为每个外部文件保留 SHA256、来源/许可、版本和放置路径的本地 manifest；不要提交该 manifest 中的私密绝对路径。

## 3. 单元测试与静态检查

从仓库根目录运行：

```bash
python -m pytest -q fold/tests
python -m compileall -q fold stage1 stage2 force_command_estimator global_anchor_force_field
git diff --check
```

纯网络、mask、轨迹处理单测不需要 ARCTIC 数据。仿真集成测试和训练必须在有许可资产及可用 NVIDIA GPU 的环境中另行验证。通过单测不代表 Genesis GPU rollout 或完整训练已复现。

## 4. 端到端流程示例（Box-200）

下面是一个与当前主流程一致的参考配置。请先把 `$DEMO_Q`、对象 URDF、锚点文件和 retarget 路径替换为已经检查过的本机输入。脚本必须从仓库根目录运行。

```bash
export ROOT="$PWD"
export PYTHONPATH="$ROOT"
export SETUPTOOLS_USE_DISTUTILS=stdlib
export WANDB_MODE=offline
export OBJECT=box
export CLIP=box-30-230-s01-use_01
export RETARGET=para
export DEMO=/path/to/FoLD/fold/assets/arctic/processed/s01/box_use_01.npy
export DEMO_Q="$ROOT/logs/local_data/box_target_q.npy"
export OBJECT_URDF="$ROOT/fold/assets/arctic/box/box.urdf"
export ANCHORS="$ROOT/global_anchor_force_field/data/box_global_anchors_200.npz"
export COMMANDS="$ROOT/force_command_estimator/data/box_gt_global_force_commands_full.npz"
export ENCODER_DIR="$ROOT/force_command_estimator/results/field_encoder/box_pointnet_300e"
mkdir -p "$(dirname "$DEMO_Q")" "$ROOT/global_anchor_force_field/data" \
  "$ROOT/force_command_estimator/data" "$ENCODER_DIR"

# 从 processed ARCTIC 序列提取 7D 物体状态；保留完整帧序列。
python -c 'import numpy as np,sys; d=np.load(sys.argv[1],allow_pickle=True).item()["params"]; q=np.concatenate((d["obj_trans"],d["obj_rot"],d["obj_arti"][:,None]),axis=1); np.save(sys.argv[2],q)' \
  "$DEMO" "$DEMO_Q"
```

### 4.1 生成 anchor 和逐帧 GT 力场命令

每台机器都应使用本机可访问的、与对象 URDF 坐标系一致的 base/lid OBJ 重新生成 anchors。ARCTIC object template 常见路径为 `$ARCTIC_ROOT/data/meta/object_vtemplates/box/{bottom,top}.obj`；先核对文件存在和 mesh 单位。以下示例按 OBJ 单位为毫米设置 `--scale 0.001`；若本机 mesh 单位不同，必须相应调整。切勿复用包含其他机器路径的 anchor 缓存。

```bash
export ARCTIC_ROOT=/path/to/authorized/arctic_data
python global_anchor_force_field/generate_global_anchors.py \
  --meshes "$ARCTIC_ROOT/data/meta/object_vtemplates/box/bottom.obj" \
           "$ARCTIC_ROOT/data/meta/object_vtemplates/box/top.obj" \
  --part-names base lid --num-anchors 200 --scale 0.001 \
  --out "$ANCHORS"
```

随后生成逐帧 GT 力场命令：

```bash
python force_command_estimator/generate_gt_commands.py \
  --anchors "$ANCHORS" --urdf "$OBJECT_URDF" --target-q "$DEMO_Q" \
  --start-frame 30 --num-frames 200 --active-top-k 60 \
  --topk-method normal_prefit --out "$COMMANDS"
```

该命令按连续 GT 状态生成每帧 command。`frame_index[i]` 的 command 对应 `GT[i] → GT[i+1]`；确认 target-q 的帧编号与 clip 约定一致，避免错一帧。

### 4.2 Stage 1 基础策略

先确认 retarget 文件存在且序列长度覆盖 clip；如需生成/修复 retarget，按 `fold/retargeting/` 工具及相应说明执行，保留 `para` 原始基线，不要覆盖。

```bash
python stage1/train_stage1.py \
  --hand inspire_hand --clip "$CLIP" -rn "$RETARGET" \
  -B 128 -am hybrid -imi 0.3 -imv 0.1 -bc 0.3 -ho 4 \
  -me 1000 -sf 1000 --overlap_geom_penalty 20 \
  --overlap_geom_threshold 0.001 --overlap_inside_backend grid \
  --overlap_grid_pitch 0.002 -exp "fold_stage1_box200"
```

在 `logs/rl_games/inspire_hand/` 下找到本次 run 的 epoch-1000 `.pth`，检查对应 `params/agent.yaml`、`params/env.pkl` 和 obs spec；明确记录 checkpoint 路径和 SHA256。以下命令假设路径保存在 `BASE_CHECKPOINT`：

```bash
export BASE_CHECKPOINT=/path/to/stage1_epoch_1000.pth
test -f "$BASE_CHECKPOINT"
sha256sum "$BASE_CHECKPOINT"
```

### 4.3 训练 128D field encoder

```bash
python force_command_estimator/train_field_encoder.py \
  --data "$COMMANDS" --epochs 300 --seed 42 --device cuda \
  --out-dir "$ENCODER_DIR"
test -f "$ENCODER_DIR/best_field_encoder.pt"
```

### 4.4 Stage 2 residual + curriculum + force command

```bash
python stage2/train_stage2.py \
  --hand inspire_hand --clip "$CLIP" -rn "$RETARGET" \
  -B 4096 -am hybrid -imi 0.3 -imv 0.1 -bc 0.3 -con 3.0 -ho 16 \
  -me 10000 -sf 1000 --base_checkpoint "$BASE_CHECKPOINT" \
  --stage2_residual_scale 0.1 --use_curriculum \
  --realtime_force_field_cmd --field_command_data "$COMMANDS" \
  --field_encoder_checkpoint "$ENCODER_DIR/best_field_encoder.pt" \
  --field_active_top_k 60 -exp "fold_stage2_box200_full"
```

本发布版本的正式流程使用全局 anchor 选择；局部 hand-proximity mask 属于仍在验证的实验，不纳入论文匹配版本。课程学习续训不能只传 rl-games `.pth`：还必须恢复 curriculum 当前上下界和完整 gain history。发布前应提供不含机器绝对路径的课程状态恢复说明。

### 4.5 确定性完整 clip 评估

使用训练 run 保存的环境参数和选定 checkpoint；固定 seed 42、关闭对象执行器增益、同一完整帧范围及成功阈值。评估入口参数随版本可能变化，先执行 `python fold/rl/eval_rl_games.py --help`。当前入口支持 `--object_control_mode zero`、`--seed 42`、`--position_success_threshold 0.15`、`--rotation_success_threshold_deg 40`、`--env_params_from`、`--eval_method` 和 `--checkpoint_role`。示例：

```bash
python fold/rl/eval_rl_games.py \
  --hand inspire_hand --clip "$CLIP" --checkpoint "$BASE_CHECKPOINT" \
  --env_params_from /path/to/stage2/params/env.pkl \
  --object_control_mode zero --seed 42 \
  --position_success_threshold 0.15 --rotation_success_threshold_deg 40 \
  --eval_task_name box-200 --eval_method ours --checkpoint_role best
```

对 Stage 2 policy，应将 `--checkpoint` 指向 Stage 2 `.pth`；Stage 1 checkpoint 仅用于训练初始化。保存完整命令、配置、checkpoint epoch/SHA256、逐帧 NPY/NPZ 和 `metrics_summary.json`。baseline 与 ours 必须使用相同 env、trajectory、seed 和终止规则，不能把训练期间的最佳 checkpoint 与另一方法的 final checkpoint 混为同一选择规则。

## 5. 完整可复现性的边界

- Python 依赖版本是参考锁定，不保证不同 GPU、驱动、Genesis kernel 或不同 ARCTIC 资产版本获得逐 bit 相同结果。
- 训练入口可运行不代表原始论文结果自动复现；必须固定资产版本、clip、seed、checkpoint selection、课程状态以及 eval 条件。
- Stage 1、force-command generation、encoder、Stage 2 和 eval 各阶段应分别保存 config、输入 hash、命令和输出 hash。
- 开源首发前，`fold/assets/` 当前已跟踪的大量第三方几何/手模型资产仍需逐项授权审核；README/requirements 的整理本身不等于发布许可审核通过。

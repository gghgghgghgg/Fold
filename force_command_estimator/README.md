# GT 力场命令与 128D Field Encoder

本目录保留当前 Stage 2 pipeline 实际使用的离线 GT command 生成与 PointNet 编码器训练代码。Stage 2 运行时会根据当前仿真状态重新求解 anchor forces，再在线编码成 128D observation；不会依赖预先导出的固定 latent。

## 核心文件

- `generate_gt_commands.py`：由物体表面 anchors、物体 GT `target_q` 和 URDF 惯量，逐帧优化 force/wrench/generalized command，并保存 anchor 位置、法向、力场和帧索引。
- `field_encoder.py`：PointNet 风格的逐 anchor 编码器与 autoencoder 模型定义。
- `train_field_encoder.py`：训练 field encoder，输出 `best_field_encoder.pt` 和归一化参数。
- `stage2/realtime_force_field.py`：Stage 2 中在线计算力场并生成 128D latent；采用全局 active top-k anchor 选择。

本目录保留当前流程必需的 GT 力场命令生成与 128D PointNet encoder 训练实现。历史 GRU 估计器、GT-reset 对比、GPU benchmark 和静态 latent 导出工具不属于当前 pipeline，当前候选中不提供这些工具。

## 生成 GT 力场命令

需预先准备对象 anchors、带连续物体状态的 `target_q.npy` 以及正确 URDF。帧 `i` 的输出 command 对应从 `GT[i]` 到 `GT[i+1]` 的转移。

```bash
python force_command_estimator/generate_gt_commands.py \
  --anchors global_anchor_force_field/data/box_global_anchors_200.npz \
  --urdf fold/assets/arctic/box/decomp/box_decomp.urdf \
  --target-q /path/to/authorized/box_target_q.npy \
  --start-frame 30 --num-frames 200 --active-top-k 60 \
  --topk-method normal_prefit \
  --out force_command_estimator/data/box_gt_global_force_commands_full.npz
```

完整序列可将`--num-frames`设为`0`。检查目标 qpos、anchor mesh 的坐标单位及帧编号一致；dataset 与模型权重不随源码发布。

## 训练 128D Field Encoder

```bash
python force_command_estimator/train_field_encoder.py \
  --data force_command_estimator/data/box_gt_global_force_commands_full.npz \
  --epochs 300 --seed 42 --device cuda \
  --out-dir force_command_estimator/results/field_encoder/box_pointnet_300e
```

输出的`best_field_encoder.pt`由 Stage 2 的`--field_encoder_checkpoint`读取。

## Stage 2 在线计算

开启`--realtime_force_field_cmd`并提供`--field_command_data`和`--field_encoder_checkpoint`后，Stage 2 每个仿真步会：

1. 根据当前物体状态选择该帧 anchors，并由目标 wrench 求解 anchor forces；
2. 在全局 anchor 集合上按预拟合力大小选择 active top-k；
3. 将当前 anchors、法向和求得的 forces 输入 encoder，得到 128D policy observation。

所以，force command 数据生成、field encoder 训练和 Stage 2 在线力场求解/编码是核心；离线预先导出静态 128D latent 不能替代状态相关的在线求解。

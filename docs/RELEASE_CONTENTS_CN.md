# 开源候选副本内容与排除项

> 维护者历史清单（2026-10-28）。其中的待办和验证范围仅记录撰写时状态；用户复现步骤请参见 [`REPRODUCTION_CN.md`](REPRODUCTION_CN.md) 和 [`ENVIRONMENT_SETUP_AND_REPRODUCTION.md`](ENVIRONMENT_SETUP_AND_REPRODUCTION.md)。此候选包不包含研究工作区的训练状态文档。

本目录是从当前研究工作区复制出的审阅版候选，不是 GitHub 发布、提交或授权审核完成的证明。研究工作区原代码未因本次整理而移动或删除。

## 保留内容

- Stage 1、全局 anchor 力场、128D 力场编码器、Stage 2 residual/curriculum、仿真环境与评估入口。
- 核心逻辑单测及全局 anchor 工具的 smoke-test 入口。
- 静态仿真手部与 ARCTIC 物体模型仅保留在本地候选副本，未跟踪到 Git；ARCTIC 原始下载不纳入。当前 Git 树跟踪的三条处理序列和三条retarget样例见`docs/EXAMPLE_DATA.md`。所有第三方资产的再分发权限需按其来源分别核验。
- `rl_games` 运行源码，不包含其 Git 历史、文档图片、notebook 或测试目录。

## 排除内容

- ARCTIC原始数据、额外生成的 processed/retarget 文件及接触映射缓存；当前Git树已跟踪的三条processed样例与三条对应retarget文件例外，参见`docs/EXAMPLE_DATA.md`。
- 力场 anchor/command `.npy/.npz`、encoder 权重、Stage 1/2 checkpoint、训练日志和结果。
- `analysis/`、`tools/` 中针对特定训练结果、渲染包或单次迁移编写的辅助脚本，以及队列/运维脚本。
- 真机场景中的机械臂轨迹、相机和传感器记录；这些不属于仿真策略复现所需的代码资产。
- `test_new_hand_cfg.py` 手工启动式配置检查，不作为自动化单测发布；保留对核心算法有覆盖的自动化测试。

## 发布前仍需完成

1. 核实六个已跟踪 ARCTIC 派生样例及任何拟纳入静态模型的再分发权限；Inspire 等手模型按具体来源许可证保留署名并满足相应条件。其他项目公开包含资产本身不视为 FoLD 获得同等授权。
2. 在干净克隆与目标 GPU 环境安装依赖，按 `REPRODUCTION_CN.md` 做最小 smoke test；目前候选检查仅覆盖 Python 编译和无数据单测。
3. 审查 Git 忽略规则、候选文件清单与历史记录；区分已跟踪的六个样例和其他本地数据，排除生成缓存、checkpoint、真机场景采集数据和机器路径。
4. 补充版本化外部数据清单与校验和格式；不得将个人机器绝对路径或真实场景采集数据加入公开包。

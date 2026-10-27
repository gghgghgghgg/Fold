# GitHub 开源整理清单（工作稿）

## 代码范围

建议首个可审查源码版本保留：

- `stage1/`：Stage 1 网络、观测规格与训练入口。
- `force_command_estimator/`：力场命令生成、求解、PointNet 编码及训练代码；不包含 `data/`、`results/` 中的 NPZ、checkpoint 和实验输出。
- `stage2/`：残差网络、实时 wrench/force-field 命令、观测拼接和训练入口（论文匹配版使用全局 anchor top-k）。
- `fold/`：上游仿真/重定向代码中本项目实际依赖的模块，以及经许可核实可发布的配置和资产。
- `analysis/`、`logs/`、`tools/`：只选择通用的数据检查、绘图、验证和可复现脚本；逐个审查后再纳入，不整目录上传。
- 根目录环境说明、安装指南、源码许可证与上游论文引用。

## 不应作为源码发布的本地内容

当前 `.gitignore` 排除日志、报告、渲染视频、checkpoint、数组数据和本地模型资产。策略复现所需的仿真机械手模型与配置应在许可允许时纳入；当前候选 Git 快照先不提交待审资产。真机场景机械臂轨迹、相机/传感器记录等数据不属于代码发布内容。
发布前还必须核查**已跟踪文件**：`.gitignore` 不会自动从 Git 历史或索引中移除既有文件。尤其检查：

- 真机场景机械臂轨迹、相机/传感器采集、真实场景录制及硬件标定数据：不纳入 GitHub；
- 仿真机械手模型/URDF/mesh/纹理和配置：FoLD流程复现所需，保留已授权的必要版本；手模型文件的许可证仍须确认；
- `fold/assets/` 中的 ARCTIC 物体模型、纹理、碰撞网格、URDF，以及新增的 Genesis 兼容物体资产；
- 任何真实演示/重定向轨迹、用户提供资产、个人路径或机器配置；
- 第三方库的复制代码、预训练权重、模型文件和生成数据；
- Git 历史中曾提交过、后来才加入 ignore 的大文件或敏感文件。

这些内容在权利人、数据许可和发布授权未逐项确认前，不应随源码仓库公开。真机场景采集数据不随GitHub发布；必需的仿真手模型则在授权允许时保留。可将受限运行数据列为外部依赖清单，提供放置路径，但不得重新打包上传。

## 发布前阻断项

1. 明确最终 GitHub 目标仓库。当前本地 remote 指向 `gghgghgghgg/fold-private` 与上游 `MandiZhao/fold`，尚未指向此前创建的 `gghgghgghgg/force-cmd`；本工作稿不会改 remote、提交或推送。
2. 对所有随仓库发布的源码和资产确认版权、数据条款与许可兼容性。根目录 MIT 文件不自动授权 ARCTIC 或其他第三方数据/模型。
3. 基于 `git ls-files` 和 Git 历史生成逐文件发布清单；`.gitignore` 对已跟踪文件无效，因此要保留必要仿真手模型、排除真机场景机械臂数据，并确认目标仓库历史没有意外携带训练/采集数据。
4. 在干净 clone 中验证依赖安装、入口导入、CPU 单元测试；有 GPU 时再验证一个短 smoke run。注明专有/定制 Genesis 与 rl-games 依赖及版本。
5. 补足方法说明、命令行示例、输入输出 schema、checkpoint 来源和局限；不要把本地绝对路径或训练产物写入 README。
6. 由仓库维护者确认作者、单位/职务成果披露要求、许可证和首次公开时间后，才做公开切换或 push。

## Deferred experimental features

The hand-proximity/local-anchor mask is excluded from this paper-matched release snapshot pending validation. It may be added in a later version with its own training/evaluation results and clear separation from the global-anchor reference method.

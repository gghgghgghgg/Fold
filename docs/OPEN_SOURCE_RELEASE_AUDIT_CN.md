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

DexMachina README 声明其已发布论文使用的灵巧手与 ARCTIC 资产，但这不自动授予 FoLD 镜像和再发布第三方文件的权利。ARCTIC 官方许可证明确把模型/3D网格纳入受限 Data & Software，并禁止未经书面许可向第三方提供。当前 `.gitignore` 排除整个 `fold/assets/`；在获得适用于 FoLD 发布的许可前，不将这些资产加入索引。训练数据、日志、checkpoint、processed/retargeted 序列和真机场景采集记录始终排除。
发布前还必须核查**已跟踪文件**：`.gitignore` 不会自动从 Git 历史或索引中移除既有文件。尤其检查：

- 真机场景机械臂轨迹、相机/传感器采集、真实场景录制及硬件标定数据：不纳入 GitHub；
- 仿真机械手模型/URDF/mesh/纹理和配置：未取得适用于 FoLD 再发布的来源许可前不纳入；
- `fold/assets/arctic/` 中可能受 ARCTIC 许可约束的静态物体模型/网格：未经权利人书面许可不纳入；
- 任何真实演示/重定向轨迹、用户提供资产、个人路径或机器配置；
- 第三方库的复制代码、预训练权重、模型文件和生成数据；
- Git 历史中曾提交过、后来才加入 ignore 的大文件或敏感文件。

真机场景采集数据不随 GitHub 发布。ARCTIC 演示、模型、处理序列与 retarget 结果按其许可边界处理；DexMachina 的公开仓库可作为用户获取其资产版本的来源，但 FoLD 不因此自动获得镜像权。发布前检查实际跟踪清单，确保没有误纳入未获授权的资产或实验产物。

## 发布前阻断项

1. 确认最终 GitHub 目标仓库并审查该远端现有历史；本地 remote 当前指向 `gghgghgghgg/Fold`，尚未推送本候选仓库。
2. 对计划纳入的每个静态模型取得适用于 FoLD 的授权，并记录原始提供方、许可证和修改内容；ARCTIC 书面许可未确认前不得纳入其模型/网格。
3. 基于 `git ls-files` 和 Git 历史生成逐文件发布清单；确认获授权资产之外，ARCTIC 序列、生成 arrays、训练产物和真机场景机械臂数据均被排除。
4. 在干净 clone 中验证依赖安装、入口导入、CPU 单元测试；有 GPU 时再验证一个短 smoke run。注明专有/定制 Genesis 与 rl-games 依赖及版本。
5. 补足方法说明、命令行示例、输入输出 schema、checkpoint 来源和局限；不要把本地绝对路径或训练产物写入 README。
6. 由仓库维护者确认作者、单位/职务成果披露要求、许可证和首次公开时间后，才做公开切换或 push。

## Deferred experimental features

The hand-proximity/local-anchor mask is excluded from this paper-matched release snapshot pending validation. It may be added in a later version with its own training/evaluation results and clear separation from the global-anchor reference method.

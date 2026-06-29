# AMD ROCm 策略复刻专题

本专题面向希望在 AMD Ryzen AI MAX+ / Radeon GPU 设备上复刻 LeRobot、ACT、SmolVLA 和 pi_0 的小伙伴。它不是单纯的环境安装笔记，而是一套围绕“复刻是否真的成功”的组队学习实践：大家会从设备资源检查开始，逐步完成 MuJoCo 抓杯任务的数据审计、ACT 闭环诊断、SmolVLA 加权采样、pi_0 权限 smoke test 和实验报告整理。

学完本专题后，大家应当能够：

- 判断 AMD ROCm 设备是否具备训练和推理条件；
- 区分显存、统一内存、温度、风扇模式和训练稳定性的关系；
- 用 `physical_success` 复核策略是否真的夹起杯子，而不是只满足几何成功；
- 解释 ACT 在闭环部署中为什么会失败，以及 DAgger / oracle correction 解决了什么；
- 用红杯、蓝杯固定指令评估 SmolVLA 是否存在任务分布偏置；
- 在 pi_0 训练前完成 Hugging Face gated model 权限检查和 1-step smoke；
- 把训练日志、成功率表格和代表视频整理成别人能读懂、能复现实验判断的报告。

## 适合谁学习

本专题适合已经完成 Every Embodied 基础章节，并希望在国产或异构 GPU 环境中做真实复刻的小伙伴。大家最好已经了解：

- Python / conda / uv 的基础环境管理；
- LeRobot 数据集的基本结构；
- MuJoCo 中 observation、action、rollout 的含义；
- ACT、SmolVLA、pi_0 的大致区别。

如果大家还没有跑过原始 MuJoCo 教程，建议先学习：

- [LeRobot MuJoCo 训练 ACT、SmolVLA、pi_0 教程](../../06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA/README.md)
- [策略诊断与物理成功评估](../../06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA/09策略诊断与物理成功评估.md)

## 章节目录

| 任务 | Markdown 概述 | Notebook 实操 |
| --- | --- | --- |
| 01 | [AMD ROCm 设备与环境确认](./README_01_AMD_ROCm设备与环境确认.md) | [01_device_env_check.ipynb](./notebooks/01_device_env_check.ipynb) |
| 02 | [物理成功评估与视频复核](./README_02_物理成功评估与视频复核.md) | [02_physical_success_review.ipynb](./notebooks/02_physical_success_review.ipynb) |
| 03 | [ACT 在 ROCm 上的迁移与 DAgger 诊断](./README_03_ACT_ROCm迁移与DAgger诊断.md) | [03_act_dagger_diagnostics.ipynb](./notebooks/03_act_dagger_diagnostics.ipynb) |
| 04 | [SmolVLA 在 ROCm 上的迁移与采样加权](./README_04_SmolVLA_ROCm迁移与采样加权.md) | [04_smolvla_weighted_sampling.ipynb](./notebooks/04_smolvla_weighted_sampling.ipynb) |
| 05 | [pi_0 权限 smoke 与训练门控](./README_05_pi0_ROCm权限Smoke与训练门控.md) | [05_pi0_smoke_gate.ipynb](./notebooks/05_pi0_smoke_gate.ipynb) |
| 06 | [ROCm 调试复盘与排障案例](./README_06_ROCm调试复盘与排障案例.md) | [06_rocm_debug_playbook.ipynb](./notebooks/06_rocm_debug_playbook.ipynb) |

Markdown 章节主要负责讲清楚背景、判断口径和实验结论；Notebook 负责逐格运行检查、读取指标、生成图表和整理命令模板。大家学习时可以先读 Markdown，再打开对应 Notebook 跟着跑。

## 阶段性复刻状态

本专题的示例实验中，ACT 和 SmolVLA 已经形成了完整的训练、评估和视频复核链路；pi_0 还停留在权限、下载和 smoke test 门控阶段。大家阅读时可以把它当成三种不同层级的复刻状态：SmolVLA 是相对稳定的结果案例，ACT 是典型的闭环诊断案例，pi_0 是大模型策略训练前的准备案例。

![当前复刻状态总览](./assets/model_status_summary.png)

图 1：本专题示例实验的阶段性状态。大家需要注意，这里使用的是更严格的 `physical_success`：ACT 和 SmolVLA 已经有批量评估与代表视频，pi_0 还需要先通过稳定的 1-step smoke，再进入正式训练。

## 推荐学习节奏

| 任务 | 建议时长 | 主要产出 |
| --- | --- | --- |
| Task 1 | 0.5 天 | AMD 设备资源表、ROCm 检查日志、缓存目录规划 |
| Task 2 | 0.5 天 | `physical_success` 评估脚本、成功/失败视频样例 |
| Task 3 | 1 天 | ACT open-loop / closed-loop 诊断表、DAgger 数据设计 |
| Task 4 | 1 天 | SmolVLA 红/蓝杯成功率对照表、加权采样实验 |
| Task 5 | 0.5 到 1 天 | pi_0 smoke test、权限排障记录、最终报告模板 |
| Task 6 | 0.5 天 | 排障复盘、失败案例、实验报告整理 |

## Notebook 还是 Python 脚本

本专题建议同时保留两类材料：

| 形式 | 适合内容 | 原因 |
| --- | --- | --- |
| Notebook | 环境检查、单条 rollout 可视化、教学解释 | 方便大家逐格观察状态、图像和动作 |
| Python 脚本 | 批量评估、严格成功率、批量视频录制、训练入口 | 结果更可复现，也适合远端 AMD 设备长时间运行 |

建议不要把所有诊断都塞进 Notebook。批量评估和训练入口应该脚本化，这样组队学习时不同同学的结果更容易比较。

## 学习产物怎么整理

完成复刻后，大家不要只留下一串命令或一段“跑通了”的描述。更好的做法是把证据整理成一份小型实验报告，让别人能看出你验证了什么、还没有验证什么。

一份合格的实验报告至少包含：

| 资料 | 作用 | 建议写法 |
| --- | --- | --- |
| 环境表 | 说明实验在哪类硬件和 ROCm 版本上完成 | 写 GPU/APU 型号、系统、ROCm、PyTorch、温度和显存占用 |
| 数据表 | 说明训练数据是否可信 | 写 episode 数量、任务类型、红/蓝杯比例、是否通过物理回放审计 |
| 成功率表 | 说明模型是否真的完成任务 | 同时写 `legacy_success` 和 `physical_success`，优先解释后者 |
| 代表视频 | 帮助大家理解成功和失败行为 | 至少放 1 个真实成功和 1 个典型失败，配关键帧或图注 |
| 排障记录 | 说明为什么这样修 | 按“现象、证据、根因、修复、验证”整理，不贴长日志 |
| 命令模板 | 帮助别人复现 | 使用 `$PROJECT_ROOT`、`$DATA_ROOT`、`$OUTPUT_ROOT` 这类变量 |

结果报告不需要包含模型权重、缓存目录、完整训练日志或个人机器路径。大家只需要保留足够复现实验判断的内容：命令模板、短日志片段、summary 表格、关键视频和清楚的结论。

## 最小成果模板

完成本专题后，建议每位小伙伴整理一份结果摘要：

| 项目 | 内容 |
| --- | --- |
| 设备 | AMD GPU / APU 型号、ROCm 版本、系统版本 |
| 数据 | episode 数量、任务类型、红/蓝杯比例 |
| ACT | best checkpoint、严格成功率、主要失败类型 |
| SmolVLA | 红杯成功率、蓝杯成功率、采样策略 |
| pi_0 | 是否通过 gated 权限、是否通过 1-step smoke |
| 视频 | 1 个真实成功、1 个典型失败 |
| 复盘 | 这次复刻中最关键的一个坑 |

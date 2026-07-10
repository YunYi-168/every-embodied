# Notebook 实操入口

本目录保存 AMD ROCm 策略复刻专题的配套 Notebook。01–06 负责环境、指标和诊断；07–11 补齐从键盘采集、正式训练到 MuJoCo closed-loop 的完整执行链。Markdown 章节负责讲清楚概念、判断口径和实验结论，Notebook 负责逐格运行代码、生成配置、启动任务和整理结果表。

建议从专题根目录启动 Jupyter：

```bash
cd /path/to/04-AMD-ROCm策略复刻专题
jupyter lab
```

如果在自己的 AMD 设备或远端服务器上运行，请先按实际情况设置：

```bash
export PROJECT_ROOT=/path/to/every-embodied/mujoco_pnp
export DATA_ROOT=/path/to/datasets/every_embodied
export OUTPUT_ROOT=/path/to/outputs
export MODEL_ROOT="$PROJECT_ROOT/ckpt"
```

| Notebook | 对应章节 | 主要用途 |
| --- | --- | --- |
| [01_device_env_check.ipynb](./01_device_env_check.ipynb) | 01 设备与环境确认 | 检查 ROCm、PyTorch、显存、温度和目录规划 |
| [02_physical_success_review.ipynb](./02_physical_success_review.ipynb) | 02 物理成功评估 | 理解 `physical_success`，复核成功/失败关键帧 |
| [03_act_dagger_diagnostics.ipynb](./03_act_dagger_diagnostics.ipynb) | 03 ACT 诊断 | 查看 ACT 进展曲线，整理 DAgger 评估命令 |
| [04_smolvla_weighted_sampling.ipynb](./04_smolvla_weighted_sampling.ipynb) | 04 SmolVLA 加权采样 | 比较红/蓝杯成功率，重新生成图表 |
| [05_pi0_smoke_gate.ipynb](./05_pi0_smoke_gate.ipynb) | 05 pi_0 训练门控 | 检查 gated 权限、1-step smoke 和训练命令模板 |
| [06_rocm_debug_playbook.ipynb](./06_rocm_debug_playbook.ipynb) | 06 排障复盘 | 按“现象、证据、根因、修复、验证”整理问题 |
| [07_data_collection_and_audit.ipynb](./07_data_collection_and_audit.ipynb) | 07 数据采集 | 键盘采集红/蓝杯 LeRobot 数据，并拦截推杯误判 |
| [08_act_training_rocm.ipynb](./08_act_training_rocm.ipynb) | 08 ACT 训练 | 生成配置，完成 2-step smoke 和 5,000 步基线训练 |
| [09_smolvla_training_rocm.ipynb](./09_smolvla_training_rocm.ipynb) | 09 SmolVLA 训练 | 完成基础权重加载、smoke 和分指令训练准备 |
| [10_pi0_training_rocm.ipynb](./10_pi0_training_rocm.ipynb) | 10 pi_0 训练 | 检查 gated 权限，完成 smoke 和正式训练 |
| [11_mujoco_closed_loop_deploy.ipynb](./11_mujoco_closed_loop_deploy.ipynb) | 11 闭环部署 | 在 MuJoCo 中运行 checkpoint，保存 JSONL、视频和严格成功率 |

第一次从零学习时，推荐顺序是 `01 → 07 → 08 → 11 → 02/03`。ACT 闭环跑通后，再执行 `09 → 11 → 04` 和 `10 → 11 → 05/06`。如果已经有数据和 checkpoint，可以直接从 02–06 学习诊断。

07 的交互采集和 11 的可视化 rollout 需要可用的 `DISPLAY`；08–10 的长训练需要足够的模型缓存、checkpoint 空间和稳定电源。所有 `RUN_*` 开关默认关闭，先检查命令和路径，再显式打开。

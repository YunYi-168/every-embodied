# Notebook 实操入口

本目录保存 AMD ROCm 策略复刻专题的配套 Notebook。Markdown 章节负责讲清楚概念、判断口径和实验结论；Notebook 负责逐格检查环境、读取指标、展示图片、生成命令模板和整理结果表。

建议大家从专题根目录启动 Jupyter：

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

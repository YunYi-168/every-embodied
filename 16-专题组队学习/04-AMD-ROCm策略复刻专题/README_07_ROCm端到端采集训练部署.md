# ROCm 端到端采集、训练与 MuJoCo 部署

前六个任务重点讲设备检查、物理成功评估和模型诊断。本章把此前缺少的基础执行链补齐：从键盘示教开始，生成 LeRobot 数据集，在 AMD ROCm 上训练 ACT、SmolVLA 和 pi_0，再把 checkpoint 放回 MuJoCo 做闭环评估。

这里的“跑通”包含五个连续检查点：

1. MuJoCo viewer 能接收键盘操作并完成一条真实抓取；
2. LeRobot 数据集包含正确的图像、状态、动作和语言指令；
3. 2-step smoke 能完成数据加载、前向、反向和 checkpoint 写出；
4. 正式训练能稳定使用 AMD GPU，不出现 OOM、NaN 或 kernel 退出；
5. checkpoint 在未见 seed 的 MuJoCo closed-loop 中通过 `physical_success`，并有视频可以复核。

训练 loss 下降、open-loop 动作误差变小或者脚本没有报错，都不能单独证明第五项成立。

## 学习路线

| 阶段 | Notebook | 主要产物 |
| --- | --- | --- |
| 数据采集 | [07_data_collection_and_audit.ipynb](./notebooks/07_data_collection_and_audit.ipynb) | 双相机、6D state、7D action、红/蓝杯指令的 LeRobot 数据集 |
| ACT 训练 | [08_act_training_rocm.ipynb](./notebooks/08_act_training_rocm.ipynb) | ACT smoke 与正式 checkpoint |
| SmolVLA 训练 | [09_smolvla_training_rocm.ipynb](./notebooks/09_smolvla_training_rocm.ipynb) | SmolVLA smoke 与正式 checkpoint |
| pi_0 训练 | [10_pi0_training_rocm.ipynb](./notebooks/10_pi0_training_rocm.ipynb) | gated 权限检查、pi_0 smoke 与正式 checkpoint |
| 闭环部署 | [11_mujoco_closed_loop_deploy.ipynb](./notebooks/11_mujoco_closed_loop_deploy.ipynb) | 固定 seed 成功率、JSONL 指标和 rollout 视频 |

第一次学习时建议先用 ACT 完成整条链路。ACT 模型较小，训练速度快，数据或动作定义有问题时更容易定位。ACT 闭环通过后，再在同一份语言数据上训练 SmolVLA，最后处理 pi_0 的 gated 权限、模型加载和小数据接触控制问题。

## 两种起点

### 从零开始

依次运行 07、08 和 11，先得到一个 ACT 闭环结果；随后运行 09、10，再回到 11 切换模型评估。这条路线不要求事先准备训练好的 checkpoint。

### 已有数据或 checkpoint

如果已经完成上游 `04mujoco复现ACT、Pi0、SmolVLA`，可以设置下面的环境变量，直接进入对应训练或部署 Notebook：

```bash
export PROJECT_ROOT=/path/to/every-embodied/06-策略抓取或抓取VLA/大模型控制、VLA、VLM/04mujoco复现ACT、Pi0、SmolVLA
export DATA_ROOT=/path/to/large-disk/datasets/every_embodied
export MODEL_ROOT=/path/to/large-disk/checkpoints/every_embodied
export OUTPUT_ROOT=/path/to/large-disk/outputs/every_embodied
```

`PROJECT_ROOT` 保存轻量源码；数据、checkpoint、视频和缓存应放在容量充足的磁盘。不要把这些大文件提交到教程仓库。

## 数据采集边界

采集与 GPU 厂商无关。只要 MuJoCo、LeRobot 版本、场景 XML、控制频率、state/action schema 和成功判定一致，就可以在另一台带 NVIDIA GPU 或只有 CPU 的桌面机器采集，再把整个数据集目录同步到 AMD 训练机。

07 Notebook 对上游交互采集做了三项修正：

- 每条轨迹保存后关闭记录开关，不把 reset 过程混进下一条 episode；
- 红杯和蓝杯指令交替出现，减少小数据任务分布偏斜；
- 只有杯子真实抬升、保持直立并完成释放时才保存，避免“把杯子推到盘子上”被当成成功示教。

远端 JupyterHub 或 Code Server 如果无法让 MuJoCo viewer 获得键盘焦点，不要在无显示会话里硬采。可以使用远程桌面，也可以在本地采集后同步数据。

## smoke 与正式训练

08–10 都会生成两份 YAML：

- smoke 配置只运行 2 步，用于验证链路；
- full 配置使用该模型的基线训练步数，并保存中间 checkpoint。

Notebook 中的 `RUN_SMOKE` 和 `RUN_FULL_TRAIN` 默认都是 `False`。先检查数据路径、输出目录和生成配置，再显式打开。smoke 通过后还要确认：

- `rocm-smi` 中 GPU 利用率和显存占用合理；
- loss 与梯度范数是有限值；
- checkpoint 写到 `$MODEL_ROOT`；
- 系统盘仍有足够空间；
- 没有把 Hugging Face token 写进 Notebook 或日志。

pi_0 还需要 Hugging Face gated model 权限。Notebook 只检查 `HF_TOKEN` 是否存在，不会打印或保存 token。

## MuJoCo closed-loop

11 Notebook 会按 20 Hz 执行下面的闭环：

```text
双相机图像 + 6D 关节状态 + 语言指令
                    ↓
              ACT / SmolVLA / pi_0
                    ↓
          7D 关节目标与夹爪命令
                    ↓
                 MuJoCo
                    ↓
              下一帧真实观测
```

评估默认只预览配置，不会自动加载大模型。确认 `POLICY_TYPE`、`MODEL_RUN_DIR`、数据统计和显示会话后，将 `RUN_CLOSED_LOOP=True`。小面板可以先跑 4 个 seed，但最终结论至少需要 20–30 个 held-out seeds，并分别报告红杯和蓝杯。

输出目录包含：

- `results.jsonl`：每个 seed 的 `legacy_success`、`physical_success`、lift、xy 和 upright；
- `<policy>_seed<seed>.mp4`：用于复核真实行为的 rollout 视频；
- 终端汇总：严格成功数和总 episode 数。

如果 `legacy_success` 高于 `physical_success`，说明出现了推杯、空抓、运输中掉落或其它几何误判。此时回到任务 02–06，按 observation、action、接触阶段和视频证据继续诊断。

## 完成标准

完成这一章后，实验记录至少应包含：

| 项目 | 最低要求 |
| --- | --- |
| 环境 | AMD 设备、ROCm、PyTorch 与显存/温度记录 |
| 数据 | episode 数、红蓝杯分布、state/action shape、物理回放审计 |
| smoke | 2-step 正常完成，checkpoint 可读取 |
| 正式训练 | 训练步数、batch size、中间 checkpoint 和资源曲线 |
| 部署 | held-out seeds、`physical_success`、一个成功视频和一个失败视频 |
| 结论 | 明确区分 raw policy、辅助 head 和带脚手架的 hybrid 结果 |

这份结果可以继续接入 ACT DAgger、SmolVLA Weighted sampler 和 pi_0 尾段诊断，不需要重新搭建基础链路。

# 05 pi_0 权限 smoke 与训练门控

本任务关注 pi_0。pi_0 的复刻难点不只是训练本身，还包括 gated model 权限、模型权重下载、缓存管理和大模型初始化。建议大家先完成 smoke test，再决定是否启动长训练。

配套实操 Notebook：[05_pi0_smoke_gate.ipynb](./notebooks/05_pi0_smoke_gate.ipynb)。

## 先检查 Hugging Face 权限

pi_0 依赖 PaliGemma。大家需要确认：

1. Hugging Face 账号已接受 `google/paligemma-3b-pt-224` 的 gated 条款；
2. token 具备 public gated repository 读取权限；
3. 远端机器可以访问 Hugging Face；
4. token 没有出现在命令行、Notebook 输出或日志里。

推荐脚本形态：

```bash
HF_TOKEN_STDIN=1 REQUIRE_PROXY=1 ./install_hf_token_for_pi0.sh
```

这个脚本应该从隐藏输入或 stdin 读取 token，先验证 `whoami`、PaliGemma 和 `lerobot/pi0`，全部通过后再保存 token。

## 1-step smoke test

权限通过后，先跑 1-step smoke：

```bash
RUN_SMOKE=1 RUN_FULL_TRAIN=0 ./run_pi0_train_eval_after_hf_ready.sh
```

这个 smoke test 证明：

- gated model 权限可用；
- 数据集能加载；
- pi_0 policy 能构造；
- 至少一次 forward/backward/optimizer step 能跑通；
- checkpoint 保存链路正常。

它不证明策略已经收敛，也不代表最终成功率。

## 正式训练门控

只有当 smoke test 通过后，才启动正式训练：

```bash
RUN_SMOKE=1 RUN_FULL_TRAIN=1 PI0_STEPS=20000 PI0_BATCH_SIZE=4 ./run_pi0_train_eval_after_hf_ready.sh
```

正式训练前建议确认：

| 检查项 | 状态 |
| --- | --- |
| PaliGemma 权限 | 通过 |
| `lerobot/pi0` 权限 | 通过 |
| 1-step smoke | 通过 |
| GPU 温度 | 可接受 |
| VRAM / 内存 | 有余量 |
| checkpoint 目录 | 空间足够 |
| 代理或网络 | 稳定 |

## 常见问题

| 报错 | 含义 | 处理 |
| --- | --- | --- |
| 401 Unauthorized | token 无效或未登录 | 重新生成 token |
| 403 Forbidden / gated | token 或账号没有 gated 权限 | 接受模型条款，打开 public gated repo access |
| 下载卡住 | 代理、Xet 或网络不稳定 | 保留缓存后重试，必要时禁用 Xet |
| GPU 未启动 | 仍在下载或初始化模型 | 观察进程、缓存和日志 |
| OOM | batch 太大或统一内存压力过高 | 降 batch size，关无关进程 |

## 权限、缓存和日志的安全习惯

pi_0 会下载较大的 gated model 权重，调试时也会涉及 Hugging Face token。这里的重点不是“把命令跑起来”就结束，而是让整个训练链路可控、可复查、可继续。

建议大家形成下面几个习惯：

- token 只通过隐藏输入、环境变量或 Hugging Face CLI 管理，不复制到 Notebook markdown 和训练日志里；
- Hugging Face cache 放在空间充足的缓存目录，不放在项目源码目录；
- 权重下载失败时，先保留已有缓存，再检查网络、代理和 gated 权限；
- 训练日志中只长期保存关键摘要，例如模型是否加载成功、是否完成 1-step smoke、checkpoint 是否写出；
- 记录命令时用 `$PROJECT_ROOT`、`$HF_HOME`、`$OUTPUT_ROOT` 这类变量，方便大家换机器复现。

完成权限和缓存检查后，建议把结果写成一张小表：

| 检查项 | 应观察到什么 | 如果失败先查什么 |
| --- | --- | --- |
| `whoami` | 能识别当前 Hugging Face 账号 | token 是否有效 |
| PaliGemma config | 能读取 gated model 配置 | 是否接受模型条款、token 权限是否打开 |
| `lerobot/pi0` | 能读取策略配置或权重索引 | 网络、代理、HF cache |
| 1-step smoke | 能完成一次 forward/backward | 显存、依赖版本、数据 key |
| checkpoint 写出 | 输出目录出现 smoke checkpoint | 输出路径权限、磁盘空间 |

## 阶段性检查点

在本专题的示例实验里，pi_0 还处在训练前门控阶段：权限检查和下载链路是重点，稳定的 1-step smoke 与正式训练成功率还没有完成。因此，大家在自己的实验报告中也要把“权限通过”“smoke 通过”“正式训练完成”“策略评估完成”分开写，不要把前两步误写成策略已经复刻成功。

![当前复刻状态总览](./assets/model_status_summary.png)

图 1：pi_0 在本专题中仍处于训练门控阶段。大家可以先学习权限、缓存和 smoke test 的排障方法，等 1-step smoke 稳定后，再进入正式训练和 `physical_success` 评估。

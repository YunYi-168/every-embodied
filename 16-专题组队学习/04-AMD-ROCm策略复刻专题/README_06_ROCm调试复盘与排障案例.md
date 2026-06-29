# 06 ROCm 调试复盘与排障案例

本章把 AMD ROCm 设备上复刻 ACT、SmolVLA 和 pi_0 时遇到的关键问题整理成学习案例。它不按时间顺序罗列命令，而是把每个问题拆成一条排障链路：先观察现象，再收集证据，然后确认根因，最后用同一评估口径验证修复是否有效。

配套实操 Notebook：[06_rocm_debug_playbook.ipynb](./notebooks/06_rocm_debug_playbook.ipynb)。

学完本章后，大家应当能够：

- 判断“日志成功”和“物理成功”是否一致；
- 知道什么时候应该回到数据采集和回放审计；
- 理解为什么 ACT 要做闭环 DAgger，而不能只看训练 loss；
- 理解 SmolVLA 的颜色任务偏置为什么要用强制红/蓝指令评估；
- 在 ROCm 远端训练时定位常见的进程、显存、路径和依赖问题；
- 安全地处理 Hugging Face gated model、token 和缓存问题。

## 从现象到结论

大家自己做复刻时，也建议不要只保存一段长日志。更有价值的是把每次失败整理成可复盘的“小案例”：问题怎么发现，证据是什么，哪些方向被排除，最后怎样验证修复有效。

建议把每个问题整理成下面的形态：

| 字段 | 写什么 |
| --- | --- |
| 现象 | 大家能直接观察到的错误、视频行为或指标异常 |
| 证据 | JSONL、summary、视频关键帧、GPU 状态或短日志 |
| 根因 | 被证据支持的真实原因，不写猜测成事实 |
| 修复 | 改了哪个口径、脚本、数据策略或训练参数 |
| 验证 | 修复后用同一评估口径得到什么结果 |
| 学习结论 | 这次问题给后续复刻留下什么经验 |

命令和路径建议写成可迁移的形式，例如 `$PROJECT_ROOT`、`$DATA_ROOT`、`$OUTPUT_ROOT`。这样大家换到自己的工作站、云主机或课堂服务器时，也能照着同一套逻辑复现。

## 案例 1：旧 success 不等于真抓取成功

**现象**：ACT 有一次评估显示成功，但视频里杯子并没有被夹起，而是被末端挤到盘子附近，杯子还出现倾倒。

**证据**：旧 `env.check_success()` 更接近终态几何判断，只要杯子终态靠近盘子、夹爪打开、末端高度满足条件，就可能返回成功。视频复核发现，这种成功并不一定包含“夹起、搬运、释放、直立放置”的完整过程。

**修复**：教程里统一引入 `physical_success`，至少检查：

1. 旧几何成功条件为真；
2. 目标杯被抬起到足够高度，例如 lift 大于 `0.03 m`；
3. 抬起状态持续若干 control tick；
4. 终态杯子基本直立，例如 upright cosine 大于阈值。

**教程经验**：所有模型对比都要优先报告 `physical_success`，旧 success 可以作为辅助指标，但不能单独作为复刻成功证明。

## 案例 2：示教数据本身也要审计

**现象**：早期 ACT 训练容易学成“夹住不放”或末端释放不稳定。只看回放几何 success 时，数据似乎没有问题，但闭环 rollout 失败很多。

**根因**：早期采集脚本中，末尾夹爪 action 的记录方式有问题。它把环境返回的实际夹爪状态当成 action 写入，导致尾段标签更像“保持闭合”，而不是“执行释放”。这种错误会直接教坏策略。

**修复**：

- 修正采集脚本，保存原始控制 action 的 gripper 维度；
- 对已有数据做尾段修复；
- 训练前先对每条 episode 做 open-loop 物理回放审计，确认示教数据确实是真抓取、真放置。

**教程经验**：如果策略总是在末端阶段失败，不要先怀疑模型太弱。先检查数据 action 是否真实表达了释放动作。

## 案例 3：ACT 的 loss 下降不代表闭环稳定

**现象**：ACT 能完成训练，open-loop 或 teacher-forced 诊断也能看到一定学习效果，但 closed-loop rollout 会偏离示教状态，出现不接触杯子、抬起后倒杯、放置偏移等问题。

**排障路线**：

| 阶段 | 物理成功结果 | 学到的结论 |
| --- | --- | --- |
| clean closed-loop | 0/10 | 仅训练离线数据不能证明闭环可用 |
| timestamp offset | 3/15 | 对齐 suffix phase 后，策略开始偶发真抓取 |
| downweight DAgger | 13/30 | 降低 correction 数据权重能保护 reset-start 主分布 |
| best DAgger | 17/30 | ACT 已经形成完整诊断案例，但还不是强泛化 |

![ACT DAgger 进展曲线](./assets/act_dagger_progress_curve.png)

图 1：ACT 调试不是“多训几步”就能解决，而是要沿着数据分布、时间对齐、闭环状态纠偏逐步定位问题。

**教程经验**：ACT 的关键不是只把 checkpoint 训出来，而是证明它能在自己产生的状态分布上继续恢复任务。

## 案例 4：不是所有 DAgger 数据都会提升结果

**现象**：追加更多 DAgger 或 failure-bucket 数据后，某些分支反而退化。full-reset failure-bucket 数据直接混入主训练时，闭环成功率明显下降。

**根因**：失败状态数据和 reset-start 主分布不一定兼容。直接高权重混入，可能让模型忘掉原本已经会的起始状态行为。某些 correction suffix 还存在 timestamp/phase 与 reset-start episode 不一致的问题。

**修复策略**：

- 保护当前表现最好的 checkpoint，不让后续实验随手覆盖它；
- 新数据只在同一组 seed、同一 `physical_success` 口径下比较；
- 对 correction episode 使用 timestamp offset；
- 对 correction 数据降采样，例如 sample weight `0.25`；
- 对退化分支保留结论，但不作为主线。

**教程经验**：DAgger 的目标是补闭环跑偏后的状态，不是把所有失败轨迹都扔回训练集。新增数据必须先回答“它补的是哪个状态分布”。

## 案例 5：SmolVLA 不是不会抓，而是存在任务偏置

**现象**：SmolVLA baseline 在随机评估里看起来有一定成功率，但强制红杯和蓝杯指令后，差异非常明显：红杯可以成功，蓝杯几乎失败。

**证据**：同一批 seeds 分别固定两条指令：

```text
Place the red mug on the plate.
Place the blue mug on the plate.
```

baseline 的严格物理结果为红杯 `8/10`、蓝杯 `0/10`。数据集中红/蓝 episode 数量并不失衡，但蓝杯任务更远、更难，导致模型在蓝杯抓取前段不稳定。

![SmolVLA 红杯蓝杯固定指令成功率](./assets/smolvla_red_blue_success.png)

图 2：强制红/蓝指令评估能把“模型不会抓”和“模型对某类任务有偏置”区分开。

**教程经验**：语言策略必须按 instruction 拆开评估。只看总体成功率，很容易把颜色、距离和任务难度偏置藏起来。

## 案例 6：复制数据不一定比加权采样好

**现象**：为了修蓝杯失败，直觉做法是复制蓝杯 episode。1.5x、2x、3x 复制确实能提升蓝杯，但红杯成功率明显下降。

**结果对比**：

| 策略 | 红杯 physical success | 蓝杯 physical success | 判断 |
| --- | --- | --- | --- |
| baseline | 8/10 | 0/10 | 红杯好，蓝杯失败 |
| blue copy 1.5x | 3/10 | 7/10 | 蓝杯提升，但红杯退化 |
| blue copy 2x | 4/10 | 8/10 | 仍然偏蓝 |
| blue copy 3x | 2/10 | 8/10 | 过采样更伤红杯 |
| weighted blue 2.0 step1000 | 6/10 | 9/10 | 比复制更均衡 |
| weighted blue 2.0 step500 | 8/10 | 10/10 | 当前保护基线 |

**根因**：复制 episode 会改变数据集 episode 分布和统计，容易把模型推向某个颜色任务。Weighted sampler 不改变原始数据文件，只调整采样概率，因此更容易回滚和比较。

**教程经验**：对小数据集做任务平衡时，优先尝试 sampler/loss 层的平滑加权，再考虑复制数据。中间 checkpoint 也要评估，最终 step 不一定最好。

## 案例 7：视频是证据，但成功率以 batch JSONL 为准

**现象**：某些 seed 在 batch eval 中失败，单独复录视频时又成功；也有 seed 在 batch 中成功，单独复录时失败。

**原因**：MuJoCo rollout、渲染、控制时序和策略采样可能存在边界非确定性。单条视频非常适合教学展示，但不能替代批量 JSONL 统计。

**推荐报告方式**：

- 成功率用固定 seed batch eval；
- 成功/失败视频用于解释行为；
- 对边界失败 seed 复跑 3 次，判断是固定失败还是偶发失败；
- 教程图注要写清楚“视频是代表行为，不是成功率统计本身”。

**教程经验**：不要用一条最好看的视频替代统计表，也不要因为一条失败视频否定整个 checkpoint。二者要一起看。

## 案例 8：ROCm 远端训练要脚本化，不要全靠 Notebook

**现象**：Notebook 适合讲解，但长训练、批量评估和视频录制更容易遇到中断、路径、进程和资源问题。

本轮调试中遇到过几类典型问题：

| 问题 | 表现 | 处理 |
| --- | --- | --- |
| `PYTHONPATH` 缺失 | 临时脚本找不到项目内模块 | 在远端项目根目录设置 `PYTHONPATH=$PWD:${PYTHONPATH:-}` |
| 文件句柄泄露 | 批量 seed eval 后出现资源加载失败 | 每个 env / viewer 用完后显式关闭，必要时分批评估 |
| 训练迭代器缓存 | 长训时 RAM 不断上涨、GPU 空闲 | 不使用会缓存 batch 的无限 `cycle`，改成 iterator 用完后重建 |
| 进程残留 | GPU 或显存占用不符合预期 | 用精确命令和 PID 检查，只停止自己确认的任务 |
| 温度较高 | 长训时 GPU 持续满载 | 记录温度、显存和 fan/performance 模式；只要训练稳定且无 OOM/kernel crash，再继续比较模型结果 |

推荐大家把批量任务做成脚本，例如：

```bash
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

./.venv/bin/python eval_policy_success.py \
  --policy act \
  --act-policy-path "$MODEL_ROOT/act_best/step_5000" \
  --physical-success \
  --seed-start 1000 \
  --episodes 10 \
  --output-jsonl "$OUTPUT_ROOT/act_seen_1000_1009.jsonl"
```

**教程经验**：Notebook 用来解释和单步观察，脚本用来产生可比较结果。真正汇报成功率前，必须确认日志、JSONL、视频和 GPU 状态都能互相解释。

## 案例 9：pi_0 先过权限和 smoke，再谈训练

**现象**：pi_0 不是简单启动训练就能跑通。它依赖 gated model 权限、Hugging Face token、权重下载、缓存和大模型初始化。某些失败发生在训练之前。

**排障顺序**：

1. 确认 token 有效，但不要在命令行和日志里打印 token；
2. 确认账号已接受 PaliGemma 等 gated model 条款；
3. 检查 `lerobot/pi0` 和相关模型能否读取；
4. 确认远端网络或代理能稳定下载权重；
5. 先跑 1-step smoke；
6. smoke 通过后再启动正式训练和强制红/蓝评估。

**阶段性状态**：在本专题示例中，pi_0 仍处于训练前门控阶段。大家可以先学习权限检查、下载排障和 smoke test 流程，但不能把这些步骤等同于策略已经复刻成功。

**教程经验**：对于大模型策略，权限、下载和构造 policy 本身就是一层实验门控。没有通过 1-step smoke，就不要启动长训练。

## 最小排障记录模板

大家以后遇到新问题，可以直接复制这个模板写实验记录：

```markdown
### 问题名

- 现象：
- 复现命令：
- 关键证据：
- 排除项：
- 根因：
- 修复：
- 修复后验证：
- 学习结论：
```

其中“学习结论”最好写成一句能指导下一次实验的话，例如“旧 success 不能替代物理成功复核”或“复制 episode 会提升蓝杯但可能伤害红杯”。

## Checkpoint

完成本章后，大家应当能回答：

- 为什么本专题不直接用旧 `success` 报告结果；
- 为什么 ACT 要做 DAgger，但又不能盲目混入所有失败数据；
- 为什么 SmolVLA 要拆成红杯、蓝杯固定指令评估；
- 为什么 weighted sampler 比简单复制 episode 更适合这次蓝杯修复；
- 为什么 pi_0 暂时不能写成已完成；
- 如何把自己的调试记录整理成可复盘的实验案例，而不是照搬原始流水账。

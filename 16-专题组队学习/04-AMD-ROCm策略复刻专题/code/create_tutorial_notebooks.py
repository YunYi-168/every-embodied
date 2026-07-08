#!/usr/bin/env python3
"""Create the companion notebooks for the AMD ROCm tutorial topic."""

from __future__ import annotations

import json
from pathlib import Path


TOPIC_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = TOPIC_ROOT / "notebooks"


METADATA = {
    "kernelspec": {
        "display_name": "Python 3 (ipykernel)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
    },
}


COMMON_SETUP = r'''
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys


def find_topic_root():
    cwd = Path.cwd().resolve()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "assets" / "metrics_snapshot.json").exists():
            return candidate
    raise RuntimeError("请从 AMD ROCm 专题目录或 notebooks 子目录启动 Jupyter。")


TOPIC_ROOT = find_topic_root()
ASSET_DIR = TOPIC_ROOT / "assets"
NOTEBOOK_DIR = TOPIC_ROOT / "notebooks"
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/path/to/every-embodied/mujoco_pnp"))
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/path/to/datasets/every_embodied"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", TOPIC_ROOT / "outputs"))
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", PROJECT_ROOT / "ckpt"))

print("TOPIC_ROOT =", TOPIC_ROOT)
print("PROJECT_ROOT =", PROJECT_ROOT)
print("DATA_ROOT =", DATA_ROOT)
print("OUTPUT_ROOT =", OUTPUT_ROOT)
print("MODEL_ROOT =", MODEL_ROOT)
'''


DISPLAY_HELPERS = r'''
try:
    from IPython.display import Image, Markdown, display
except Exception:
    class Markdown(str):
        pass

    def display(obj):
        print(obj)

    def Image(filename=None, width=None):
        return f"[image] {filename}"


def show_asset(filename, width=960):
    path = ASSET_DIR / filename
    if path.exists():
        display(Image(filename=str(path), width=width))
    else:
        print(f"缺少素材：{path}")


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    display(Markdown("\n".join(lines)))
'''


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n") + "\n",
    }


def write_nb(filename: str, cells: list[dict]) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": cells,
        "metadata": METADATA,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = NOTEBOOK_DIR / filename
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(path)


NOTEBOOKS: dict[str, list[dict]] = {}


NOTEBOOKS["01_device_env_check.ipynb"] = [
    md(
        """
        # 01 AMD ROCm 设备与环境确认

        这一节的目标是先证明设备、环境、缓存和权限链路是可用的。很多训练失败看起来像模型问题，实际可能是 ROCm 没识别 GPU、缓存放错磁盘、网络无法下载权重，或者统一内存被其它进程挤占。

        Notebook 里主要做三件事：检查硬件和 PyTorch、规划大文件目录、形成一张设备资源表。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1：ROCm 和 PyTorch 是否能看到 GPU"),
    code(
        r'''
def run_cmd(cmd):
    print("$", " ".join(cmd))
    if shutil.which(cmd[0]) is None:
        print(f"未找到命令：{cmd[0]}")
        return
    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout)


run_cmd(["rocm-smi", "--showuse", "--showtemp", "--showmemuse"])

try:
    import torch
    print("torch =", torch.__version__)
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    print("torch.cuda.device_count() =", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device =", torch.cuda.get_device_name(0))
except Exception as exc:
    print("PyTorch 检查失败：", repr(exc))
'''
    ),
    md("如果 `rocm-smi` 正常、`torch.cuda.is_available()` 为 `True`，说明 ROCm 与 PyTorch 的基础链路已经打通。若前者正常但后者为 `False`，优先检查 PyTorch 是否为 ROCm build。"),
    md("## Checkpoint 2：统一内存、显存和磁盘目录"),
    code(
        r'''
paths = [
    ("项目源码", PROJECT_ROOT),
    ("数据目录", DATA_ROOT),
    ("模型与 checkpoint", MODEL_ROOT),
    ("Notebook 输出", OUTPUT_ROOT),
    ("Hugging Face cache", Path(os.environ.get("HF_HOME", "/path/to/cache/huggingface"))),
]
md_table(["项目", "建议路径", "是否存在"], [(name, f"`{path}`", path.exists()) for name, path in paths])
'''
    ),
    md("大模型缓存、数据集、checkpoint 和批量视频都可能很大。实验报告里说明目录规划和磁盘类型即可，不需要把这些文件放进教程目录。"),
    md("## Checkpoint 3：设备资源表模板"),
    code(
        r'''
rows = [
    ("设备型号", "AMD Ryzen AI MAX+ / Radeon GPU"),
    ("系统版本", "Ubuntu / WSL / 其它"),
    ("ROCm 版本", "填写 rocm-smi 或包版本"),
    ("PyTorch 版本", "填写 torch.__version__"),
    ("空闲温度", "例如 30-45 C"),
    ("训练温度", "例如 75-85 C"),
    ("ACT 训练 VRAM", "填写 rocm-smi 观察值"),
    ("SmolVLA 训练 VRAM", "填写 rocm-smi 观察值"),
    ("pi0 smoke 状态", "未跑 / 通过 / 失败原因"),
]
md_table(["项目", "记录"], rows)
'''
    ),
    md("完成本节后，应当能判断这台 AMD 设备是否已经具备继续训练和评估的条件。"),
]


NOTEBOOKS["02_physical_success_review.ipynb"] = [
    md(
        """
        # 02 物理成功评估与视频复核

        这一节解决一个关键问题：日志里的 success 是否真的代表机器人夹起杯子并放到盘子上。这里把旧的几何成功条件和 `physical_success` 分开统计，并用视频关键帧复核成功与失败。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1：读取本专题的示例指标"),
    code(
        r'''
snapshot_path = ASSET_DIR / "metrics_snapshot.json"
snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
print(json.dumps(snapshot["best_summary"], ensure_ascii=False, indent=2))
'''
    ),
    md("## Checkpoint 2：理解 `physical_success`"),
    code(
        r'''
def physical_success(row, min_lift=0.03, min_lift_steps=3, upright_threshold=0.7):
    """示例判定逻辑。实际项目中应以 eval 脚本记录的字段为准。"""
    legacy = bool(row.get("legacy_success", row.get("success", False)))
    max_lift = float(row.get("max_target_lift", row.get("max_mug_lift", 0.0)))
    lifted_steps = int(row.get("lifted_steps", 0))
    upright = float(row.get("final_target_upright_cos", row.get("final_mug_upright_cos", 1.0)))
    return legacy and max_lift >= min_lift and lifted_steps >= min_lift_steps and upright >= upright_threshold


examples = [
    {"name": "几何成功但倒杯", "success": True, "max_mug_lift": 0.08, "lifted_steps": 80, "final_mug_upright_cos": 0.2},
    {"name": "推到盘边但没夹起", "success": True, "max_mug_lift": 0.005, "lifted_steps": 0, "final_mug_upright_cos": 0.99},
    {"name": "真抓取并直立放置", "success": True, "max_mug_lift": 0.09, "lifted_steps": 120, "final_mug_upright_cos": 0.96},
]
md_table(["样例", "physical_success"], [(e["name"], physical_success(e)) for e in examples])
'''
    ),
    md("旧 success 可以作为辅助指标，但不能替代视频和物理状态复核。特别是抓杯任务里，推、挤、倒杯都可能让终态几何看起来接近成功。"),
    md("## Checkpoint 3：复核成功与失败关键帧"),
    code(
        r'''
show_asset("smolvla_blue_failure_sequence.jpg", width=1100)
show_asset("smolvla_blue_success_sequence.jpg", width=1100)
show_asset("act_failure_sequence.jpg", width=1100)
show_asset("act_success_sequence.jpg", width=1100)
'''
    ),
    md("看视频时，沿时间轴观察四件事：是否接触目标杯、是否稳定夹起、是否搬运到盘子上、终态是否直立。"),
    md("## Checkpoint 4：批量 JSONL 统计模板"),
    code(
        r'''
def summarize_jsonl(path):
    path = Path(path)
    if not path.exists():
        print(f"文件不存在：{path}")
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    total = len(rows)
    legacy = sum(bool(r.get("success") or r.get("legacy_success")) for r in rows)
    physical = sum(bool(r.get("physical_success")) for r in rows)
    return {"total": total, "legacy_success": legacy, "physical_success": physical}


example_jsonl = OUTPUT_ROOT / "eval_result.jsonl"
print("把自己的 JSONL 放到这里后运行：", example_jsonl)
print(summarize_jsonl(example_jsonl))
'''
    ),
]


NOTEBOOKS["03_act_dagger_diagnostics.ipynb"] = [
    md(
        """
        # 03 ACT 在 ROCm 上的迁移与 DAgger 诊断

        这一节用 ACT 说明：训练 loss 下降不等于闭环 rollout 成功。先看诊断曲线，再把数据回放、open-loop、closed-loop、失败视频和 DAgger 串成一条排障链路。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1：查看 ACT 诊断曲线"),
    code(
        r'''
snapshot = json.loads((ASSET_DIR / "metrics_snapshot.json").read_text(encoding="utf-8"))
rows = []
for item in snapshot["act_progress"]:
    rows.append((item["label"], f'{item["success"]}/{item["total"]}', f'{item["rate"]:.1%}'))
md_table(["阶段", "physical_success", "成功率"], rows)
show_asset("act_dagger_progress_curve.png", width=960)
'''
    ),
    md("这条曲线不是为了证明 ACT 已经完美泛化，而是为了展示排障方向：先确认数据和闭环状态，再决定是否需要 DAgger。"),
    md("## Checkpoint 2：ACT 严格评估命令模板"),
    code(
        r'''
act_policy = MODEL_ROOT / "act_best" / "step_5000"
cmd = f"""
cd "$PROJECT_ROOT"
export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
./.venv/bin/python eval_policy_success.py \\
  --policy act \\
  --act-policy-path "{act_policy}" \\
  --physical-success \\
  --seed-start 1000 \\
  --episodes 10 \\
  --max-action-steps 1200 \\
  --output-jsonl "$OUTPUT_ROOT/act_seen_1000_1009.jsonl"
"""
print(cmd)
'''
    ),
    md("正式训练或批量评估更适合在终端或后台脚本里跑。Notebook 这里负责保存命令模板、解释参数含义和读取结果。"),
    md("## Checkpoint 3：成功与失败视频对照"),
    code(
        r'''
show_asset("act_success_sequence.jpg", width=1100)
show_asset("act_failure_sequence.jpg", width=1100)
'''
    ),
    md("ACT 的失败常常不是完全不动，而是接近、夹取、搬运或释放中的某一段出问题。复核视频时要写出失败发生在哪个阶段。"),
    md("## Checkpoint 4：记录 DAgger 实验"),
    code(
        r'''
rows = [
    ("prefix 长度", "例如 40 个 control tick"),
    ("oracle 类型", "scripted policy / human correction"),
    ("timestamp offset", "例如 correction episode 使用 2.0"),
    ("采样权重", "例如 correction episode weight=0.25"),
    ("评估 seed", "seen / mixed / heldout 分开写"),
    ("替换基线条件", "必须同一 physical_success 口径超过旧基线"),
]
md_table(["记录项", "示例"], rows)
'''
    ),
]


NOTEBOOKS["04_smolvla_weighted_sampling.ipynb"] = [
    md(
        """
        # 04 SmolVLA 在 ROCm 上的迁移与采样加权

        这一节用 SmolVLA 说明：语言策略不能只看总体成功率。红杯和蓝杯指令要拆开评估，再比较复制 episode 与 Weighted sampler 的效果。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1：读取红杯/蓝杯固定指令结果"),
    code(
        r'''
snapshot = json.loads((ASSET_DIR / "metrics_snapshot.json").read_text(encoding="utf-8"))
rows = []
for item in snapshot["smolvla_forced_instruction_n10"]:
    rows.append((item["label"], item["red_text"], item["blue_text"]))
md_table(["策略", "红杯 physical_success", "蓝杯 physical_success"], rows)
show_asset("smolvla_red_blue_success.png", width=1100)
'''
    ),
    md("baseline 红杯好、蓝杯差，说明模型不是完全不会抓，而是任务条件和颜色相关分布没有学稳。"),
    md("## Checkpoint 2：复制 episode 与 Weighted sampler 的区别"),
    code(
        r'''
rows = [
    ("复制 episode", "直接改变数据集 episode 分布", "实现简单，但容易让模型偏向某个颜色"),
    ("Weighted sampler", "不改原始 parquet，只改采样概率", "更容易回滚，也更适合小数据平衡"),
    ("中间 checkpoint 评估", "不要只看 final step", "本示例里 step500 比 step1000 更均衡"),
]
md_table(["方法", "做法", "为什么要注意"], rows)
'''
    ),
    md("## Checkpoint 3：重新生成图表"),
    code(
        r'''
cmd = [
    sys.executable,
    str(TOPIC_ROOT / "code" / "generate_tutorial_assets.py"),
    "--source-root",
    str(OUTPUT_ROOT),
]
print("如果 OUTPUT_ROOT 中有本章需要的 TSV/JSONL/MP4，可以运行：")
print(" ".join(cmd))
# subprocess.run(cmd, check=True)
'''
    ),
    md("上面的单元默认不直接执行，避免输出目录还没准备好时生成占位图。确认 `$OUTPUT_ROOT` 正确后，取消最后一行注释即可。"),
    md("## Checkpoint 4：蓝杯失败和修复后成功对照"),
    code(
        r'''
show_asset("smolvla_blue_failure_sequence.jpg", width=1100)
show_asset("smolvla_blue_success_sequence.jpg", width=1100)
'''
    ),
]


NOTEBOOKS["05_pi0_smoke_gate.ipynb"] = [
    md(
        """
        # 05 pi_0 权限 smoke 与训练门控

        这一节不急着启动长训练，而是先检查 PaliGemma、Hugging Face gated 权限、权重下载、数据加载和 1-step smoke。只有 smoke 通过后，正式训练才有意义。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1：权限检查命令模板"),
    code(
        r'''
print("""
cd "$PROJECT_ROOT"
HF_TOKEN_STDIN=1 REQUIRE_PROXY=1 ./install_hf_token_for_pi0.sh
""")
'''
    ),
    md("这个命令形态强调两件事：token 不写进 Notebook，脚本先验证 `whoami`、PaliGemma 和 `lerobot/pi0`，全部通过后再保存。"),
    md("## Checkpoint 2：1-step smoke 证明什么"),
    code(
        r'''
rows = [
    ("gated model 权限", "能读取 PaliGemma / pi0 所需配置"),
    ("数据加载", "能读取 LeRobot 数据集和 observation/action key"),
    ("policy 构造", "pi0 policy 能初始化"),
    ("一次训练步", "forward/backward/optimizer step 能跑通"),
    ("checkpoint 写出", "输出目录出现 smoke checkpoint"),
]
md_table(["检查项", "通过后说明什么"], rows)
'''
    ),
    md("1-step smoke 不证明策略收敛，也不代表成功率。它只是正式训练前的门槛。"),
    md("## Checkpoint 3：训练命令模板"),
    code(
        r'''
print("""
cd "$PROJECT_ROOT"
RUN_SMOKE=1 RUN_FULL_TRAIN=1 PI0_STEPS=20000 PI0_BATCH_SIZE=4 \\
  ./run_pi0_train_eval_after_hf_ready.sh
""")
'''
    ),
    md("正式训练建议放到终端、tmux 或后台脚本中执行。Notebook 负责记录参数、解释门控和读取训练后的 summary。"),
    md("## Checkpoint 4：阶段性状态图"),
    code('show_asset("model_status_summary.png", width=900)'),
    md("## Checkpoint 5：`tcp_to_plate` 后段 finisher 诊断"),
    md(
        """
        raw pi_0 目前还不能写成复刻成功。更有价值的一轮诊断是把后段 finisher 的 state 从 19 维扩到 22 维，额外加入 `tcp_to_plate = tcp_link_xyz - plate_xyz`。这一步让模型显式知道盘心相对 TCP 的方向，避免后段只靠图像和 phase 隐式猜目标。

        这个实验仍然是诊断 scaffold：前缀由 oracle 提供，schedule 从 `move_preplace` 对齐，finisher 使用 22D state。后续 30-seed 排查发现，短 `move_preplace` schedule 会提前 lower / release；强制使用长 schedule 模板后，改好版 scaffold 能稳定通过 30 个未见 seed。这个结果证明瓶颈被定位和修复，但不代表 raw pi_0 已经端到端成功。

        下一步开始拆 scaffold。第一个被替换的是手写 gripper 规则：用 phase-only logistic head 预测夹爪开闭，EEF/arm 仍由 22D finisher 输出。full-state gripper head 虽然训练集准确率 `100%`，但闭环上线失败；phase-only head 只看 `timestamp + phase_index_norm + phase_onehot11`，在完整 unseen seeds `1010-1039` 上仍保持 strict `30/30`。这个对照说明，小数据短事件 head 的输入要克制，不要把闭环会漂移的机械臂状态全塞进去。

        第二块脚手架先用固定阈值 adaptive gate 替代强制长 `move_preplace` 模板，完整 unseen seeds `1010-1039` 仍是 strict `30/30`，但其中一部分依赖 `max_steps=180` 安全兜底。继续往前走，我们训练了 logistic transition head。第一版 full head 训练集准确率约 `99.27%`，但 seed `1010` 闭环早放失败；第二版只保留 `tcp_to_plate_xy + local_step_norm`，训练集准确率约 `95.01%`，却在完整 unseen seeds `1010-1039` 上达到 strict `30/30`、legacy `30/30`。30 条里 `29/30` 是 transition head 主动触发，只有 seed `1021` 走到 max-step 安全兜底。这个对照说明，小数据 phase head 的输入也要克制，稳定几何线索比训练集高准确率更重要。

        第三层继续拆前段 contact primitive。naive all-head 把所有 transition 都按 phase tail 学，训练集指标不差，但 smoke `1010/1011` 变成 `0/2`，因为 `descend_to_close` 会在 TCP 还高出抓取 floor 约 `0.08 m` 时提前触发。修复版把 `pregrasp_to_descend` 改成几何标签，特征加入 TCP 到 grasp / pregrasp / floor 的相对量，并在部署时加 `descend_floor_guard`。完整 unseen seeds `1010-1039` 为 strict `30/30`、legacy `30/30`；核心切换 `pregrasp->descend`、`descend->close`、`close->lift` 都是 `30/30` 由 head 触发，floor guard 共阻挡 `342` 次高空 close。这仍是 contact scaffold，不是 raw pi_0 端到端成功。

        第四层拆掉 `dataset_schedule` 尾段，改成 `dynamic_timed` finisher。第一次完整 30 seed 只有 strict `27/30`，失败 seeds `1021/1031/1036` 看起来像固定 dwell 不合适；复盘代码后发现根因是 `--tcpplate-prefix-target-state` 这个全局开关越界了：prefix 需要 `tcp_to_target`，但 dynamic finisher 也误用了它，而 finisher 训练时应该吃 `tcp_to_plate`。把 state builder 改成 stage-aware 后，prefix 阶段用 target-relative，finisher 阶段用 plate-relative；三个失败 seed 复测 `3/3`，旧版同一环境连续跑完整 unseen seeds `1010-1039` 回到 strict `30/30`、legacy `30/30`。但继续 trace 又发现，连续评估时前一个 episode 的 MuJoCo qvel / ctrl / free-joint 动态残留可能污染下一个 seed，seed `1035` 曾出现采样范围外的初始位置。于是 evaluator 增加 `--fresh-env-per-episode` 和 `--hard-reset-sim-data`。最终用 `--hard-reset-sim-data` clean protocol 重跑完整 unseen seeds `1010-1039`，仍是 strict `30/30`、legacy `30/30`，红杯 `18/18`、蓝杯 `12/12`；mean `xy=0.0219 m`，max `xy=0.0450 m`，最小 lift `0.1093 m`，最小 upright cos `0.9504`。之前 seed `1036` 的 `0.0993 m` 更像 reset 残留造成的评估伪影，不再作为边界样本处理。
        """
    ),
    code(
        r'''
rows = [
    ("raw full20 open-loop", "1/20 final，3/20 ever", "raw policy 仍未过关"),
    ("raw closed-loop strict", "0/20", "最接近部署口径"),
    ("template-tail full20", "4/20", "固定尾段只能救部分样本"),
    ("旧 19D finisher handoff", "prefix 120/180/240/300 均为 0/2", "后段目标仍偏在杯子侧"),
    ("22D tcp_to_plate finisher", "strict 5/10，legacy 6/10", "plate-relative state 是有效线索，但仍是 scaffold"),
    ("22D + phase-scripted gripper", "strict 7/10，legacy 7/10", "夹爪时序能救一批失败，剩余主要是红杯落点"),
    ("unseen 30，schedule 0..9 重复", "strict 21/30，legacy 23/30", "失败全部来自短 move_preplace 模板"),
    ("unseen 30，强制长 schedule episode 0", "strict 30/30，legacy 30/30", "改好版 scaffold 稳定，但仍不是 raw pi_0"),
    ("长 schedule + phase-only learned gripper head", "strict 30/30，legacy 30/30", "第一块脚手架被可学习 head 替代"),
    ("schedule 0..9 + adaptive move gate + learned gripper", "strict 30/30，legacy 30/30", "第二块脚手架工程版：不再强制长 episode0"),
    ("schedule 0..9 + xy-step transition head + learned gripper", "strict 30/30，legacy 30/30", "第二块脚手架可学习版：29/30 主动触发"),
    ("policy prefix + contact transition heads + floor guard", "strict 30/30，legacy 30/30", "第三层去脚手架：核心接触切换由 head 触发，但仍保留 scaffold"),
    ("stage-aware dynamic_timed finisher + hard-reset clean protocol", "strict 30/30，legacy 30/30", "第四层去脚手架：prefix 用 target，finisher 用 plate；mean xy=0.0219 m，max xy=0.0450 m"),
]
md_table(["口径", "结果", "怎么理解"], rows)
'''
    ),
    code(
        r'''
print("""
cd "$PROJECT_ROOT"

# 1. 把 19D phase-state 数据转换成 22D tcp_to_plate 数据。
python code/pi0/build_lerobot_state_phase_tcpplate.py \\
  --src-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_eef_abs_10eps_v1 \\
  --src-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_eef_abs_10eps_v1 \\
  --dst-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --dst-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --overwrite

# 2. 评估 22D finisher。注意：这个命令需要已有 22D checkpoint。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1000,1001,1002,1003,1004,1005,1006,1007,1008,1009 \\
  --finisher-phase-schedule-episodes 0,1,2,3,4,5,6,7,8,9 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval/summary.json"

# 3. 更大样本量复核时，可以强制使用长 move_preplace 模板。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-force-schedule-episode 0 \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_long_schedule/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_long_schedule/summary.json"

# 4. 训练 phase-only gripper head。这个 head 只替代 gripper 规则，不改 EEF/arm。
python code/pi0/train_tcpplate_gripper_head.py \\
  --dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --feature-mode phase \\
  --label-source action \\
  --output "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --summary-json "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg_summary.json"

# 5. 用 learned gripper head 复核 30 个 unseen seed。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-force-schedule-episode 0 \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_gripper_head/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_gripper_head/summary.json"

# 6. 去掉强制长 episode0，改用 adaptive move_preplace gate。
# 这个版本使用 schedule 0..9 正常重复；如果 live TCP-to-plate 还太远，
# 就 hold move_preplace，直到 xy 足够小或达到保守 max-step 兜底。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-adaptive-move-preplace \\
  --tcpplate-adaptive-move-preplace-xy-threshold 0.05 \\
  --tcpplate-adaptive-move-preplace-min-steps 20 \\
  --tcpplate-adaptive-move-preplace-max-steps 180 \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --finisher-phase-schedule-episodes 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_adaptive_gate/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_adaptive_gate/summary.json"

# 7. 训练 xy-step transition head。full-state transition head 训练集更准，
# 但会被高度/方向相关性误导；稳定版只用 tcp_to_plate_xy + local_step_norm。
python code/pi0/train_tcpplate_transition_head.py \\
  --dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --feature-mode xy_step \\
  --step-scale 180 \\
  --output "$MODEL_ROOT/pi0_tcpplate_finisher/transition_head_tcpplate_xy_step_logreg.npz" \\
  --summary-json "$MODEL_ROOT/pi0_tcpplate_finisher/transition_head_tcpplate_xy_step_logreg_summary.json"

# 8. 用 learned transition head 替代固定阈值 gate，schedule 0..9 正常重复。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-transition-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/transition_head_tcpplate_xy_step_logreg.npz" \\
  --tcpplate-transition-head-threshold 0.5 \\
  --tcpplate-adaptive-move-preplace-min-steps 20 \\
  --tcpplate-adaptive-move-preplace-max-steps 180 \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --prefix-source oracle \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --finisher-phase-schedule-episodes 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_transition_head/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_tcpplate_eval_unseen30_transition_head/summary.json"

# 9. 训练 target-relative contact transition heads。
# naive all-head 会学到“阶段快结束了”，不等于“现在可以安全闭爪”；
# 这里用 pregrasp geometry label + grasp/floor geometry features。
python code/pi0/train_tcptarget_contact_transition_head.py \\
  --dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --task-set all \\
  --early-label-mode pregrasp_geometry \\
  --feature-mode grasp_geom_count_step \\
  --step-scale 180 \\
  --output-npz "$MODEL_ROOT/pi0_tcpplate_finisher/contact_transition_head_pregraspgeom_graspgeom_step.npz" \\
  --summary-json "$MODEL_ROOT/pi0_tcpplate_finisher/contact_transition_head_pregraspgeom_graspgeom_step_summary.json"

# 10. 用 contact transition heads 替代前段 pregrasp/descend/close 的部分手写切换。
# descend_floor_guard 仍保留，防止 TCP 还明显高于抓取 floor 时提前闭爪。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-schedule-start-phase move_preplace \\
  --tcpplate-prefix-target-state \\
  --tcpplate-contact-primitive guided_contact_lift \\
  --tcpplate-contact-transition-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/contact_transition_head_pregraspgeom_graspgeom_step.npz" \\
  --tcpplate-contact-transition-head-threshold 0.5 \\
  --tcpplate-contact-pregrasp-hold 999 \\
  --tcpplate-contact-descend-hold 999 \\
  --tcpplate-contact-transition-head-strict \\
  --tcpplate-contact-descend-floor-guard \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --tcpplate-transition-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/transition_head_tcpplate_xy_step_logreg.npz" \\
  --tcpplate-adaptive-move-preplace-min-steps 20 \\
  --tcpplate-adaptive-move-preplace-max-steps 180 \\
  --prefix-source policy \\
  --prefix-policy-path "$MODEL_ROOT/pi0_prefix_tcptarget/checkpoints/001000/pretrained_model" \\
  --prefix-dataset-repo-id datawhale_eai_pnp_fullprefix_oracle_prefix_until_mpreplace_state_phase_tcptarget_eef_abs_10eps_v1 \\
  --prefix-dataset-root ./demo_data_pi0_fullprefix_oracle_prefix_until_mpreplace_state_phase_tcptarget_eef_abs_10eps_v1 \\
  --prefix-steps 180 \\
  --finisher-phase-state dataset_schedule \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --finisher-phase-schedule-episodes 0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_contact_transition_head_unseen30/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_contact_transition_head_unseen30/summary.json"

# 11. 拆掉 dataset_schedule 尾段，改用 stage-aware dynamic_timed finisher。
# 关键点：--tcpplate-prefix-target-state 只让 prefix 使用 tcp_to_target；
# stage-aware evaluator 会让 finisher 回到 tcp_to_plate。
PYTHONPATH="$PWD:$TOPIC_ROOT/code/pi0:${PYTHONPATH:-}" \\
python code/pi0/evaluate_pi0_two_stage_tcpplate.py \\
  --tcpplate-base-evaluator code/pi0/evaluate_pi0_two_stage_eef_abs.py \\
  --tcpplate-prefix-target-state \\
  --tcpplate-contact-primitive guided_contact_lift \\
  --tcpplate-contact-transition-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/contact_transition_head_pregraspgeom_graspgeom_step.npz" \\
  --tcpplate-contact-transition-head-threshold 0.5 \\
  --tcpplate-contact-pregrasp-hold 999 \\
  --tcpplate-contact-descend-hold 999 \\
  --tcpplate-contact-transition-head-strict \\
  --tcpplate-contact-descend-floor-guard \\
  --tcpplate-gripper-head-path "$MODEL_ROOT/pi0_tcpplate_finisher/gripper_head_phase_logreg.npz" \\
  --prefix-source policy \\
  --prefix-policy-path "$MODEL_ROOT/pi0_prefix_tcptarget/checkpoints/001000/pretrained_model" \\
  --prefix-dataset-repo-id datawhale_eai_pnp_fullprefix_oracle_prefix_until_mpreplace_state_phase_tcptarget_eef_abs_10eps_v1 \\
  --prefix-dataset-root ./demo_data_pi0_fullprefix_oracle_prefix_until_mpreplace_state_phase_tcptarget_eef_abs_10eps_v1 \\
  --prefix-steps 180 \\
  --finisher-phase-state dynamic_timed \\
  --finisher-start-phase move_preplace \\
  --timed-phase-dwell-spec move_preplace:260,lower_to_plate:40,retreat:40 \\
  --hard-reset-sim-data \\
  --finisher-scripted-gripper \\
  --finisher-policy-path "$MODEL_ROOT/pi0_tcpplate_finisher/checkpoints/000250/pretrained_model" \\
  --finisher-dataset-repo-id datawhale_eai_pnp_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --finisher-dataset-root ./demo_data_pi0_dagger_prefix120_step1500_state_phase_tcpplate_eef_abs_10eps_v1 \\
  --seeds 1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1031,1032,1033,1034,1035,1036,1037,1038,1039 \\
  --output-jsonl "$OUTPUT_ROOT/pi0_dynamic_timed_stageaware_unseen30/results.jsonl" \\
  --summary-json "$OUTPUT_ROOT/pi0_dynamic_timed_stageaware_unseen30/summary.json"
""")
'''
    ),
    md("如果要跑夹爪脚本化对照，在评估命令里额外加入 `--finisher-scripted-gripper`。这一项只改 gripper，不改 EEF/arm，用来判断开闭爪时序是不是主要瓶颈。`--tcpplate-force-schedule-episode 0` 用于复核长 `move_preplace` 模板是否能消掉提前释放问题。`--tcpplate-gripper-head-path` 会在 `--finisher-scripted-gripper` 的钩子里加载 learned head，把手写 phase 规则替换成可训练的小模型。`--tcpplate-adaptive-move-preplace` 把强制长模板换成 live progress gate；`--tcpplate-transition-head-path` 则把固定阈值 gate 换成 logistic transition head；`--tcpplate-contact-transition-head-path` 进一步替代前段 contact primitive 的部分 phase 切换。`dynamic_timed` 版本要特别注意 stage-aware state：prefix 可以用 `tcp_to_target`，finisher 必须用 `tcp_to_plate`。批量评估建议加 `--hard-reset-sim-data`，先清掉跨 episode 的 MuJoCo 动力学残留；`--fresh-env-per-episode` 更干净但反复创建环境时可能遇到资产 provider 偶发报错。当前 stage-aware dynamic finisher 在 hard-reset clean protocol 的 30 个 unseen seed 上是 strict `30/30`，但仍保留 target-relative contact scaffold、floor guard、learned gripper head 和后段 finisher。"),
    md("如果旧 checkpoint 的 state normalizer 是 19 维，而新数据是 22 维，会出现 `mean/std` shape mismatch。诊断实验里可以复制 checkpoint 并删除 `normalize_inputs.buffer_observation_state.mean/std`，让 22D 数据集统计重新初始化；不要把这个临时处理误写成模型结构创新。"),
]


NOTEBOOKS["06_rocm_debug_playbook.ipynb"] = [
    md(
        """
        # 06 ROCm 调试复盘与排障案例

        这一节把复刻中的折腾整理成排障工作簿。不需要记住所有命令，关键是把一个失败现象拆成证据、根因、修复和验证。
        """
    ),
    code(COMMON_SETUP),
    code(DISPLAY_HELPERS),
    md("## Checkpoint 1：排障记录模板"),
    code(
        r'''
template = """
### 问题名

- 现象：
- 复现命令：
- 关键证据：
- 排除项：
- 根因：
- 修复：
- 修复后验证：
- 学习结论：
"""
print(template)
'''
    ),
    md("## Checkpoint 2：九个典型案例索引"),
    code(
        r'''
rows = [
    ("旧 success 误判", "必须引入 physical_success"),
    ("示教数据 gripper 标签问题", "先审计数据，再怀疑模型"),
    ("ACT loss 下降但闭环失败", "open-loop 与 closed-loop 分开看"),
    ("DAgger 数据混入后退化", "新增数据要说明补的是哪个状态分布"),
    ("SmolVLA 红蓝杯偏置", "按 instruction 拆开评估"),
    ("复制数据伤害另一类任务", "优先试 sampler/loss 层加权"),
    ("视频与 batch 统计不一致", "视频解释行为，成功率看 batch JSONL"),
    ("ROCm 长训练资源问题", "脚本化、分批评估、检查进程和温度"),
    ("pi0 训练前门控", "权限和 smoke 通过后再谈长训练"),
    ("pi0 reset 残留", "同一环境连续评估前先清掉 MuJoCo 动力学状态"),
]
md_table(["案例", "学习结论"], rows)
'''
    ),
    md("## Checkpoint 3：pi0 hard-reset 评估协议案例"),
    md(
        """
        `dynamic_timed` finisher 的一次排障很典型：stage-aware state 修复后，旧版连续环境评估已经能跑到 strict `30/30`，但 seed `1036` 的终态 `xy` 一度贴近阈值。继续 trace 时发现，后一个 seed 的初始杯子位置偶尔跑出该 seed 的采样范围，说明前一个 episode 的 qvel / ctrl / free-joint 动态状态污染了下一个 episode。

        这里不要急着把它归因成模型泛化问题。更干净的做法是先修评估协议：小面板可以用 `--fresh-env-per-episode` 每个 seed 新建环境；完整批量评估推荐 `--hard-reset-sim-data`，在每次 `env.reset(seed)` 前先清掉底层 MuJoCo sim data。用 hard-reset clean protocol 重跑 unseen seeds `1010-1039` 后，结果仍是 strict `30/30`、legacy `30/30`，mean `xy=0.0219 m`，max `xy=0.0450 m`，seed `1036` 不再贴边。
        """
    ),
    code(
        r'''
rows = [
    ("现象", "旧连续环境评估 30/30，但 seed 1036 一度 max xy=0.0993 m"),
    ("异常证据", "seed 1035 初始杯子位置出现采样范围外的值"),
    ("根因", "env.reset(seed) 重设物体位置，但没有先清掉底层 MuJoCo 动力学残留"),
    ("修复", "新增 --fresh-env-per-episode 和 --hard-reset-sim-data"),
    ("clean 结果", "hard-reset 后 1010-1039 为 strict 30/30，mean xy=0.0219 m，max xy=0.0450 m"),
]
md_table(["项", "记录"], rows)
'''
    ),
    md("## Checkpoint 4：把指标和视频放在一起看"),
    code(
        r'''
show_asset("act_dagger_progress_curve.png", width=900)
show_asset("smolvla_red_blue_success.png", width=1000)
'''
    ),
    md("图表回答“整体表现如何”，视频回答“行为到底像不像成功”。这两类证据最好同时放进实验报告。"),
    md("## Checkpoint 5：结果摘要模板"),
    code(
        r'''
rows = [
    ("设备", "GPU/APU 型号、ROCm、PyTorch、温度和显存"),
    ("数据", "episode 数量、任务类型、是否通过物理回放审计"),
    ("ACT", "best checkpoint、physical_success、主要失败类型"),
    ("SmolVLA", "红杯/蓝杯成功率、采样策略、保护基线"),
    ("pi0", "gated 权限、1-step smoke、正式训练状态"),
    ("视频", "一个真实成功、一个典型失败"),
    ("复盘", "最关键的一个坑和修复证据"),
]
md_table(["项目", "应该写什么"], rows)
'''
    ),
]


def main() -> None:
    for filename, cells in NOTEBOOKS.items():
        write_nb(filename, cells)


if __name__ == "__main__":
    main()

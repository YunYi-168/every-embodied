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

        大家在 Notebook 里主要做三件事：检查硬件和 PyTorch、规划大文件目录、形成一张设备资源表。
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
    md("大模型缓存、数据集、checkpoint 和批量视频都可能很大。大家只需要在报告里说明目录规划和磁盘类型，不需要把这些文件放进教程目录。"),
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
    md("完成本节后，大家应当能判断这台 AMD 设备是否已经具备继续训练和评估的条件。"),
]


NOTEBOOKS["02_physical_success_review.ipynb"] = [
    md(
        """
        # 02 物理成功评估与视频复核

        这一节解决一个关键问题：日志里的 success 是否真的代表机器人夹起杯子并放到盘子上。大家会看到为什么要把旧的几何成功条件和 `physical_success` 分开统计，并用视频关键帧复核成功与失败。
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
    md("看视频时，大家要沿时间轴观察四件事：是否接触目标杯、是否稳定夹起、是否搬运到盘子上、终态是否直立。"),
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

        这一节用 ACT 说明：训练 loss 下降不等于闭环 rollout 成功。大家会先看诊断曲线，再学习如何把数据回放、open-loop、closed-loop、失败视频和 DAgger 串成一条排障链路。
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
    md("ACT 的失败常常不是完全不动，而是接近、夹取、搬运或释放中的某一段出问题。大家复核视频时要写出失败发生在哪个阶段。"),
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

        这一节用 SmolVLA 说明：语言策略不能只看总体成功率。大家会把红杯和蓝杯指令拆开评估，再比较复制 episode 与 Weighted sampler 的效果。
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
    md("上面的单元默认不直接执行，避免大家还没准备好输出目录时生成占位图。确认 `$OUTPUT_ROOT` 正确后，取消最后一行注释即可。"),
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
]


NOTEBOOKS["06_rocm_debug_playbook.ipynb"] = [
    md(
        """
        # 06 ROCm 调试复盘与排障案例

        这一节把复刻中的折腾整理成排障工作簿。大家不用记住所有命令，但要学会把一个失败现象拆成证据、根因、修复和验证。
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
]
md_table(["案例", "学习结论"], rows)
'''
    ),
    md("## Checkpoint 3：把指标和视频放在一起看"),
    code(
        r'''
show_asset("act_dagger_progress_curve.png", width=900)
show_asset("smolvla_red_blue_success.png", width=1000)
'''
    ),
    md("图表回答“整体表现如何”，视频回答“行为到底像不像成功”。这两类证据最好同时放进实验报告。"),
    md("## Checkpoint 4：结果摘要模板"),
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

#!/usr/bin/env python3
"""Generate small tutorial figures from ROCm reproduction outputs.

Pass the local experiment output root with --source-root when regenerating
contact sheets from videos or refreshing metrics from JSONL/TSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency for keyframes
    cv2 = None


HERE = Path(__file__).resolve().parent
TOPIC_ROOT = HERE.parent
ASSET_DIR = TOPIC_ROOT / "assets"

FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

SMOLVLA_COMPARE_REL = (
    "smolvla_weighted_blue2_step500_forced_red_blue_20260629_1550/"
    "compare_baseline_weightedblue2_step500_step1000_blue15x_blue2x_blue3x.tsv"
)

SMOLVLA_FALLBACK = [
    ("baseline", 0.80, 0.00, "8/10", "0/10"),
    ("copy 1.5x", 0.30, 0.70, "3/10", "7/10"),
    ("copy 2x", 0.40, 0.80, "4/10", "8/10"),
    ("copy 3x", 0.20, 0.80, "2/10", "8/10"),
    ("weighted 1k", 0.60, 0.90, "6/10", "9/10"),
    ("weighted 500", 0.80, 1.00, "8/10", "10/10"),
]

ACT_STAGE_FILES = [
    (
        "clean closed-loop",
        [
            "act_clean_v2_gripper_bce05cont_eval_step5000/"
            "closedloop_trainish_seeds_1000_1004_physical.jsonl",
            "act_clean_v2_gripper_bce05cont_eval_step5000/"
            "closedloop_general_seeds_1030_1034_physical.jsonl",
        ],
        0,
        10,
    ),
    (
        "time offset",
        [
            "act_reset_oracle_v1/plus_prefix40_stable61_toffset2_eval_step5000/"
            "eval_seen_1000_1004_clampts.jsonl",
            "act_reset_oracle_v1/plus_prefix40_stable61_toffset2_eval_step5000/"
            "eval_mixed_1030_1034_clampts.jsonl",
            "act_reset_oracle_v1/plus_prefix40_stable61_toffset2_eval_step5000/"
            "eval_heldout_1080_1084_clampts.jsonl",
        ],
        3,
        15,
    ),
    (
        "downweight DAgger",
        [
            "act_reset_oracle_v1/downweight025_toffset2_stable61_eval/"
            "eval_step5000_expand10/eval_seen_1000_1009_clampts.jsonl",
            "act_reset_oracle_v1/downweight025_toffset2_stable61_eval/"
            "eval_step5000_expand10/eval_mixed_1030_1039_clampts.jsonl",
            "act_reset_oracle_v1/downweight025_toffset2_stable61_eval/"
            "eval_step5000_expand10/eval_heldout_1080_1089_clampts.jsonl",
        ],
        13,
        30,
    ),
    (
        "best DAgger",
        [
            "act_reset_oracle_v1/dagger_best025_eval_step5000_expand10/"
            "seen_1000_1009.jsonl",
            "act_reset_oracle_v1/dagger_best025_eval_step5000_expand10/"
            "mixed_1030_1039.jsonl",
            "act_reset_oracle_v1/dagger_best025_eval_step5000_expand10/"
            "heldout_1080_1089.jsonl",
        ],
        17,
        30,
    ),
]

VIDEO_SPECS = [
    (
        "smolvla_blue_failure_sequence.jpg",
        "SmolVLA baseline blue failure",
        "smolvla_step5000_blue_failure_video_20260629_101547/"
        "seed0_forced_blue_failure.mp4",
    ),
    (
        "smolvla_blue_success_sequence.jpg",
        "SmolVLA weighted blue success",
        "smolvla_weighted_blue2_step500_representative_videos_20260629_1626/"
        "videos/seed29_blue_success.mp4",
    ),
    (
        "act_failure_sequence.jpg",
        "ACT DAgger typical failure",
        "act_reset_oracle_v1/dagger_best025_representative_videos/seed1088.mp4",
    ),
    (
        "act_success_sequence.jpg",
        "ACT DAgger physical success",
        "act_reset_oracle_v1/dagger_best025_representative_videos/seed1089.mp4",
    ),
]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(path, size)


def percent_text(value: float) -> str:
    return f"{round(value * 100):.0f}%"


def count_physical_success(paths: Iterable[Path]) -> tuple[int, int] | None:
    success = 0
    total = 0
    found_any = False
    for path in paths:
        if not path.exists():
            continue
        found_any = True
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total += 1
                if bool(row.get("physical_success")):
                    success += 1
    if not found_any or total == 0:
        return None
    return success, total


def load_act_stages(source_root: Path | None) -> list[dict]:
    stages = []
    for label, rels, fallback_success, fallback_total in ACT_STAGE_FILES:
        measured = None
        if source_root is not None:
            measured = count_physical_success(source_root / rel for rel in rels)
        success, total = measured or (fallback_success, fallback_total)
        stages.append(
            {
                "label": label,
                "success": success,
                "total": total,
                "rate": success / total if total else 0.0,
            }
        )
    return stages


def load_smolvla_compare(source_root: Path | None) -> list[dict]:
    compare_path = source_root / SMOLVLA_COMPARE_REL if source_root else None
    if compare_path is None or not compare_path.exists():
        return [
            {
                "label": label,
                "red_rate": red,
                "blue_rate": blue,
                "red_text": red_text,
                "blue_text": blue_text,
            }
            for label, red, blue, red_text, blue_text in SMOLVLA_FALLBACK
        ]

    label_map = {
        "baseline_step5000": "baseline",
        "blue15x_copy": "copy 1.5x",
        "blue2x_copy": "copy 2x",
        "blue3x_copy": "copy 3x",
        "weighted_blue2_step1000": "weighted 1k",
        "weighted_blue2_step500": "weighted 500",
    }
    grouped: dict[str, dict] = {}
    with compare_path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            variant = row["variant"]
            label = label_map.get(variant, variant)
            entry = grouped.setdefault("label:" + label, {"label": label})
            color = row["color"]
            entry[f"{color}_rate"] = float(row["physical_success_rate"])
            entry[f"{color}_text"] = row["physical_success"]

    order = [item[0] for item in SMOLVLA_FALLBACK]
    return sorted(grouped.values(), key=lambda x: order.index(x["label"]))


def draw_axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = box
    axis = "#39434d"
    grid = "#d8dee6"
    font = load_font(18)
    for i in range(6):
        value = i / 5
        y = y1 - int((y1 - y0) * value)
        draw.line((x0, y, x1, y), fill=grid, width=1)
        draw.text((x0 - 62, y - 11), percent_text(value), fill=axis, font=font)
    draw.line((x0, y0, x0, y1), fill=axis, width=2)
    draw.line((x0, y1, x1, y1), fill=axis, width=2)


def draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str) -> None:
    draw.text((52, 30), title, fill="#18202a", font=load_font(34, bold=True))
    draw.text((52, 76), subtitle, fill="#55606d", font=load_font(20))


def save_grouped_bars(data: list[dict], out_path: Path) -> None:
    width, height = 1280, 760
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "SmolVLA red/blue forced-instruction success",
        "physical_success; n=10 per color for each variant",
    )

    chart = (118, 150, 1220, 600)
    draw_axes(draw, chart)
    x0, y0, x1, y1 = chart
    n = len(data)
    group_w = (x1 - x0) / n
    bar_w = 46
    red = "#d84a4a"
    blue = "#2878d7"
    text_font = load_font(18)
    label_font = load_font(17)

    for i, row in enumerate(data):
        center = x0 + group_w * (i + 0.5)
        for offset, color, key, text_key in [
            (-bar_w * 0.62, red, "red_rate", "red_text"),
            (bar_w * 0.62, blue, "blue_rate", "blue_text"),
        ]:
            value = row[key]
            bx0 = int(center + offset - bar_w / 2)
            bx1 = int(center + offset + bar_w / 2)
            by1 = y1
            by0 = y1 - int((y1 - y0) * value)
            draw.rounded_rectangle((bx0, by0, bx1, by1), radius=4, fill=color)
            draw.text(
                (bx0 - 8, by0 - 28),
                row.get(text_key, percent_text(value)),
                fill="#18202a",
                font=text_font,
            )

        label = row["label"]
        label_w = draw.textlength(label, font=label_font)
        draw.text((center - label_w / 2, y1 + 20), label, fill="#26313d", font=label_font)

    legend_y = 646
    draw.rounded_rectangle((118, legend_y, 145, legend_y + 18), radius=4, fill=red)
    draw.text((154, legend_y - 3), "red mug", fill="#26313d", font=load_font(18))
    draw.rounded_rectangle((260, legend_y, 287, legend_y + 18), radius=4, fill=blue)
    draw.text((296, legend_y - 3), "blue mug", fill="#26313d", font=load_font(18))
    draw.text(
        (52, 705),
        "Figure: weighted sampling recovers blue-mug performance without fully sacrificing red-mug performance.",
        fill="#55606d",
        font=load_font(18),
    )
    img.save(out_path)


def save_act_progress(stages: list[dict], out_path: Path) -> None:
    width, height = 1180, 720
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "ACT closed-loop physical-success progress",
        "same physical_success criterion; stages summarize iterative fixes",
    )
    chart = (120, 145, 1080, 560)
    draw_axes(draw, chart)
    x0, y0, x1, y1 = chart
    max_rate = 1.0
    color = "#20866d"
    point_fill = "#f4a340"
    line_pts = []
    font = load_font(18)
    label_font = load_font(17)

    for i, row in enumerate(stages):
        x = x0 + int((x1 - x0) * (i / max(1, len(stages) - 1)))
        y = y1 - int((y1 - y0) * (row["rate"] / max_rate))
        line_pts.append((x, y))
    draw.line(line_pts, fill=color, width=5, joint="curve")

    for i, (x, y) in enumerate(line_pts):
        row = stages[i]
        draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=point_fill, outline="#18202a", width=2)
        label = f'{row["success"]}/{row["total"]} ({percent_text(row["rate"])})'
        draw.text((x - 44, y - 44), label, fill="#18202a", font=font)
        parts = row["label"].split()
        y_text = y1 + 22
        for part in parts:
            part_w = draw.textlength(part, font=label_font)
            draw.text((x - part_w / 2, y_text), part, fill="#26313d", font=label_font)
            y_text += 22

    draw.text(
        (52, 668),
        "Figure: DAgger-style correction improves ACT, but the task still needs video-level physical review.",
        fill="#55606d",
        font=load_font(18),
    )
    img.save(out_path)


def save_best_summary(act_stages: list[dict], out_path: Path) -> None:
    rows = [
        ("ACT best DAgger", act_stages[-1]["success"], act_stages[-1]["total"], "#20866d"),
        ("SmolVLA weighted 500", 53, 60, "#2878d7"),
    ]
    width, height = 1180, 680
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    draw_header(
        draw,
        "Current reproducibility status",
        "strict physical_success; pi0 is gated-smoke pending",
    )
    chart = (120, 150, 1010, 505)
    draw_axes(draw, chart)
    x0, y0, x1, y1 = chart
    bar_w = 150
    font = load_font(20)
    label_font = load_font(19)

    for i, (label, success, total, color) in enumerate(rows):
        rate = success / total
        cx = x0 + int((x1 - x0) * (i + 1) / 3)
        by0 = y1 - int((y1 - y0) * rate)
        draw.rounded_rectangle((cx - bar_w // 2, by0, cx + bar_w // 2, y1), radius=5, fill=color)
        draw.text((cx - 50, by0 - 34), f"{success}/{total}", fill="#18202a", font=font)
        draw.text((cx - 45, by0 - 62), percent_text(rate), fill="#18202a", font=font)
        for j, part in enumerate(label.split()):
            part_w = draw.textlength(part, font=label_font)
            draw.text((cx - part_w / 2, y1 + 24 + j * 24), part, fill="#26313d", font=label_font)

    draw.rounded_rectangle((750, 238, 912, 286), radius=5, fill="#d7dce2")
    draw.text((772, 250), "pi0 pending", fill="#26313d", font=load_font(20, bold=True))
    draw.text(
        (52, 612),
        "Figure: ACT and SmolVLA have evaluated rollouts; pi0 remains pending until smoke passes.",
        fill="#55606d",
        font=load_font(18),
    )
    img.save(out_path)


def placeholder_sheet(title: str, out_path: Path, reason: str) -> None:
    width, height = 1280, 350
    img = Image.new("RGB", (width, height), "#eef1f5")
    draw = ImageDraw.Draw(img)
    draw.text((40, 42), title, fill="#18202a", font=load_font(32, bold=True))
    draw.text((40, 96), reason, fill="#55606d", font=load_font(22))
    draw.text(
        (40, 255),
        "Regenerate with: python code/generate_tutorial_assets.py --source-root /path/to/outputs",
        fill="#55606d",
        font=load_font(20),
    )
    img.save(out_path)


def sample_video_frames(video_path: Path, title: str, out_path: Path) -> None:
    if cv2 is None:
        placeholder_sheet(title, out_path, "OpenCV is not installed; video keyframes were not extracted.")
        return
    if not video_path.exists():
        placeholder_sheet(title, out_path, "Source video is unavailable in this checkout.")
        return

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if total <= 1:
        cap.release()
        placeholder_sheet(title, out_path, "Source video did not expose readable frames.")
        return

    ratios = [0.05, 0.25, 0.50, 0.75, 0.95]
    thumbs = []
    for ratio in ratios:
        idx = min(total - 1, max(0, int((total - 1) * ratio)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame)
        thumb_h = 210
        thumb_w = int(pil.width * thumb_h / pil.height)
        pil = pil.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        thumbs.append((idx / fps, pil))
    cap.release()

    if not thumbs:
        placeholder_sheet(title, out_path, "No keyframes could be decoded from the source video.")
        return

    gap = 18
    margin = 36
    title_h = 78
    label_h = 42
    content_w = sum(t.width for _, t in thumbs) + gap * (len(thumbs) - 1)
    width = max(1180, content_w + margin * 2)
    height = title_h + 210 + label_h + 28
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)
    draw.text((margin, 24), title, fill="#18202a", font=load_font(28, bold=True))

    x = margin
    y = title_h
    for seconds, thumb in thumbs:
        img.paste(thumb, (x, y))
        draw.rectangle((x, y, x + thumb.width, y + thumb.height), outline="#c5ccd6", width=2)
        label = f"t={seconds:.1f}s"
        label_w = draw.textlength(label, font=load_font(18))
        draw.text((x + thumb.width / 2 - label_w / 2, y + thumb.height + 10), label, fill="#26313d", font=load_font(18))
        x += thumb.width + gap
    img.save(out_path)


def write_metrics_snapshot(smolvla: list[dict], act_stages: list[dict], out_path: Path) -> None:
    snapshot = {
        "smolvla_forced_instruction_n10": smolvla,
        "act_progress": act_stages,
        "best_summary": {
            "act_best_dagger": {"physical_success": "17/30"},
            "smolvla_weighted_blue2_step500_expand30": {"physical_success": "53/60"},
            "pi0": "pending smoke/train verification",
        },
    }
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Local experiment outputs root. Omit it to regenerate charts from the embedded redacted metrics.",
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=ASSET_DIR,
        help="Directory for generated tutorial assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_dir = args.asset_dir
    asset_dir.mkdir(parents=True, exist_ok=True)

    source_root = args.source_root.resolve() if args.source_root else None
    smolvla = load_smolvla_compare(source_root)
    act_stages = load_act_stages(source_root)

    save_grouped_bars(smolvla, asset_dir / "smolvla_red_blue_success.png")
    save_act_progress(act_stages, asset_dir / "act_dagger_progress_curve.png")
    save_best_summary(act_stages, asset_dir / "model_status_summary.png")
    write_metrics_snapshot(smolvla, act_stages, asset_dir / "metrics_snapshot.json")

    for filename, title, rel_path in VIDEO_SPECS:
        video_path = source_root / rel_path if source_root else Path(rel_path)
        sample_video_frames(video_path, title, asset_dir / filename)


if __name__ == "__main__":
    main()

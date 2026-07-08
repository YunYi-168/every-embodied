#!/usr/bin/env python3
"""Batch replay GT or Pi0-predicted actions over multiple dataset episodes.

This is a thin, tutorial-friendly wrapper around
``replay_pi0_dataset_predictions.py``.  It keeps one Pi0 policy instance loaded
while evaluating many teacher-forced dataset episodes, then writes per-episode
JSON files plus one JSONL/summary pair.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import replay_pi0_dataset_predictions as single


def parse_episode_spec(spec: str, available: list[int]) -> list[int]:
    spec = spec.strip()
    if spec == "all":
        return available
    episodes: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            step = 1 if end >= start else -1
            episodes.extend(range(start, end + step, step))
        else:
            episodes.append(int(part))
    missing = sorted(set(episodes) - set(available))
    if missing:
        raise ValueError(f"Episodes not found in dataset: {missing}")
    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gt", "pi0"], required=True)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--repo-id", default="datawhale_eai_pnp_language")
    parser.add_argument("--dataset-root", type=Path, default=Path("./demo_data_language"))
    parser.add_argument("--stats-repo-id", default=None)
    parser.add_argument("--stats-dataset-root", type=Path, default=None)
    parser.add_argument("--episodes", default="all", help="'all', '0-19', or comma-separated episode ids.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--settle-actions", type=int, default=20)
    parser.add_argument("--max-episode-frames", type=int, default=0)
    parser.add_argument("--headless-no-viewer", action="store_true")
    parser.add_argument("--stop-on-physical-success", action="store_true")
    parser.add_argument("--reset-policy-each-frame", action="store_true")
    parser.add_argument("--clamp-action-to-episode-gt", action="store_true")
    parser.add_argument("--binarize-gripper", action="store_true")
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-open-until-step", type=int, default=-1)
    parser.add_argument("--gripper-open-tail", type=int, default=0)
    parser.add_argument("--replace-tail-with-gt", type=int, default=0)
    parser.add_argument("--append-template-tail", choices=["none", "all", "task"], default="none")
    parser.add_argument("--template-tail-steps", type=int, default=0)
    parser.add_argument("--template-blend-steps", type=int, default=0)
    parser.add_argument("--template-force-open-gripper", action="store_true")
    parser.add_argument("--physical-min-lift", type=float, default=0.03)
    parser.add_argument("--physical-min-lift-steps", type=int, default=3)
    parser.add_argument("--physical-final-upright-cos", type=float, default=0.7)
    parser.add_argument(
        "--env-create-retries",
        type=int,
        default=3,
        help="Retry MuJoCo environment creation if asset loading hits a transient filesystem/provider error.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def prepare_runtime(args: argparse.Namespace) -> None:
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.headless_no_viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")
    if args.headless_no_viewer and "pyautogui" not in sys.modules:
        sys.modules["pyautogui"] = SimpleNamespace(size=lambda: (1920, 1080))


def compute_actions(
    args: argparse.Namespace,
    dataset: Any,
    raw_indices: np.ndarray,
    policy: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, float] | None]:
    actions = []
    gt_actions = []
    pred_errors = []
    for raw_idx in raw_indices:
        item = dataset[int(raw_idx)]
        gt = single.tensor_to_np(item["action"]).reshape(-1)[:7].astype(np.float32)
        gt_actions.append(gt)
        if args.mode == "gt":
            actions.append(gt)
            continue
        if args.reset_policy_each_frame:
            policy.reset()
        batch = single.to_device_batch(item, args.device)
        with torch.no_grad():
            pred = policy.select_action(batch)[0, :7].detach().cpu().numpy().astype(np.float32)
        actions.append(pred)
        pred_errors.append(np.abs(pred - gt))

    actions_np = np.stack(actions).astype(np.float32)
    gt_np = np.stack(gt_actions).astype(np.float32)
    error_summary = None
    if pred_errors:
        err = np.stack(pred_errors)
        error_summary = {
            "mae": float(err.mean()),
            "joint_mae": float(err[:, :6].mean()),
            "gripper_abs": float(err[:, 6].mean()),
            "max_abs": float(err.max()),
        }
    return actions_np, gt_np, error_summary


def apply_postprocess(
    args: argparse.Namespace,
    dataset: Any,
    episode_column: np.ndarray,
    task: str,
    actions_np: np.ndarray,
    gt_np: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    postprocess = {
        "clamp_action_to_episode_gt": bool(args.clamp_action_to_episode_gt),
        "binarize_gripper": bool(args.binarize_gripper),
        "gripper_threshold": float(args.gripper_threshold),
        "gripper_open_until_step": int(args.gripper_open_until_step),
        "gripper_open_tail": int(args.gripper_open_tail),
        "replace_tail_with_gt": int(args.replace_tail_with_gt),
        "append_template_tail": args.append_template_tail,
        "template_tail_steps": int(args.template_tail_steps),
        "template_blend_steps": int(args.template_blend_steps),
        "template_force_open_gripper": bool(args.template_force_open_gripper),
        "stop_on_physical_success": bool(args.stop_on_physical_success),
        "template_tail_meta": None,
    }

    processed = actions_np.copy()
    if args.clamp_action_to_episode_gt:
        processed = np.clip(processed, gt_np.min(axis=0), gt_np.max(axis=0))
    if args.binarize_gripper:
        processed[:, 6] = (processed[:, 6] >= float(args.gripper_threshold)).astype(np.float32)
    if int(args.gripper_open_until_step) >= 0:
        open_steps = min(int(args.gripper_open_until_step), len(processed))
        processed[:open_steps, 6] = 1.0
    if int(args.gripper_open_tail) > 0:
        tail_open_steps = min(int(args.gripper_open_tail), len(processed))
        processed[-tail_open_steps:, 6] = 0.0
    if int(args.replace_tail_with_gt) > 0:
        tail_steps = min(int(args.replace_tail_with_gt), len(processed))
        processed[-tail_steps:] = gt_np[-tail_steps:]
    if args.append_template_tail != "none":
        template_tail, template_meta = single.build_template_tail_actions(
            dataset=dataset,
            episode_column=episode_column,
            current_task=task,
            selector=args.append_template_tail,
            tail_steps=int(args.template_tail_steps),
        )
        if args.template_force_open_gripper:
            template_tail[:, 6] = 0.0
        if int(args.template_blend_steps) > 0:
            blend_steps = int(args.template_blend_steps)
            start = processed[-1]
            end = template_tail[0]
            alphas = np.linspace(
                1.0 / float(blend_steps + 1),
                float(blend_steps) / float(blend_steps + 1),
                blend_steps,
                dtype=np.float32,
            ).reshape(-1, 1)
            blend_tail = (1.0 - alphas) * start.reshape(1, -1) + alphas * end.reshape(1, -1)
            processed = np.concatenate([processed, blend_tail.astype(np.float32), template_tail], axis=0)
        else:
            processed = np.concatenate([processed, template_tail], axis=0)
        postprocess["template_tail_meta"] = template_meta
    return processed, postprocess


def make_simple_env(args: argparse.Namespace):
    from mujoco_env.y_env2 import SimpleEnv2

    last_error: Exception | None = None
    for attempt in range(max(1, int(args.env_create_retries))):
        try:
            return SimpleEnv2("./asset/example_scene_y2.xml", action_type="joint_angle")
        except Exception as exc:  # pragma: no cover - environment dependent
            last_error = exc
            if attempt + 1 >= max(1, int(args.env_create_retries)):
                break
            print(
                f"MuJoCo env creation failed on attempt {attempt + 1}; retrying: {exc}",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(0.5)
    raise RuntimeError(f"Failed to create MuJoCo env after {args.env_create_retries} attempts") from last_error


def replay_episode(
    args: argparse.Namespace,
    dataset: Any,
    episode_column: np.ndarray,
    episode: int,
    policy: Any,
) -> dict[str, Any]:
    raw_indices = np.where(episode_column == int(episode))[0]
    if raw_indices.size == 0:
        raise ValueError(f"Episode {episode} not found")
    if args.max_episode_frames > 0:
        raw_indices = raw_indices[: int(args.max_episode_frames)]

    first_item = dataset[int(raw_indices[0])]
    task = str(first_item["task"])
    obj_init = single.tensor_to_np(first_item["obj_init"]).astype(np.float32).reshape(-1)
    if obj_init.size < 9:
        raise ValueError(f"Expected obj_init with 9 values, got {obj_init.shape}")

    if policy is not None:
        policy.reset()
    raw_actions_np, gt_np, error_summary = compute_actions(args, dataset, raw_indices, policy)
    actions_np, postprocess = apply_postprocess(args, dataset, episode_column, task, raw_actions_np, gt_np)

    env = make_simple_env(args)
    start = time.time()
    action_steps = 0
    sim_steps = 0
    success_ever = False
    physical_success_ever = False
    first_physical_success_step = None
    first_legacy_success_step = None
    try:
        env.reset(seed=0)
        env.set_instruction(task)
        env.set_obj_pose(obj_init[:3], obj_init[3:6], obj_init[6:9])
        tracker = single.init_tracker(env)
        last_debug = single.physical_debug(env, tracker, args)
        total_actions = len(actions_np) + max(int(args.settle_actions), 0)
        last_action = actions_np[0]
        while action_steps < total_actions and (args.headless_no_viewer or env.env.is_viewer_alive()):
            env.step_env()
            sim_steps += 1
            if not env.env.loop_every(HZ=args.hz):
                continue
            if action_steps < len(actions_np):
                action = actions_np[action_steps]
                last_action = action
            else:
                action = last_action
            env.step(action)
            single.update_tracker(env, tracker, args)
            last_debug = single.physical_debug(env, tracker, args)
            if last_debug["success"] and first_legacy_success_step is None:
                first_legacy_success_step = action_steps
            if last_debug["physical_success"] and first_physical_success_step is None:
                first_physical_success_step = action_steps
            success_ever = bool(success_ever or last_debug["success"])
            physical_success_ever = bool(physical_success_ever or last_debug["physical_success"])
            action_steps += 1
            if args.stop_on_physical_success and last_debug["physical_success"]:
                break
        last_debug = single.physical_debug(env, tracker, args)
    finally:
        try:
            env.env.close_viewer()
        except Exception:
            pass

    return {
        "label": args.label,
        "mode": args.mode,
        "episode": int(episode),
        "task": task,
        "num_episode_frames": int(len(raw_indices)),
        "num_actions_after_postprocess": int(len(actions_np)),
        "settle_actions": int(args.settle_actions),
        "action_steps": int(action_steps),
        "sim_steps": int(sim_steps),
        "elapsed_s": round(time.time() - start, 3),
        "success": bool(success_ever or last_debug["success"]),
        "physical_success": bool(physical_success_ever or last_debug["physical_success"]),
        "legacy_success_ever": bool(success_ever),
        "physical_success_ever": bool(physical_success_ever),
        "final_legacy_success": bool(last_debug["success"]),
        "final_physical_success": bool(last_debug["physical_success"]),
        "first_legacy_success_step": first_legacy_success_step,
        "first_physical_success_step": first_physical_success_step,
        "prediction_error": error_summary,
        "postprocess": postprocess,
        "action_stats": {
            "pred_or_replay_min": single.round_list(actions_np.min(axis=0), 5),
            "pred_or_replay_max": single.round_list(actions_np.max(axis=0), 5),
            "raw_pred_or_replay_min": single.round_list(raw_actions_np.min(axis=0), 5),
            "raw_pred_or_replay_max": single.round_list(raw_actions_np.max(axis=0), 5),
            "gt_min": single.round_list(gt_np.min(axis=0), 5),
            "gt_max": single.round_list(gt_np.max(axis=0), 5),
        },
        "debug": last_debug,
    }


def summarize(args: argparse.Namespace, results: list[dict[str, Any]]) -> dict[str, Any]:
    final_ok = sum(1 for row in results if row["final_physical_success"])
    ever_ok = sum(1 for row in results if row["physical_success_ever"])
    legacy_final_ok = sum(1 for row in results if row["final_legacy_success"])
    task_summary: dict[str, dict[str, Any]] = {}
    for row in results:
        task_row = task_summary.setdefault(
            row["task"],
            {"total": 0, "final_physical_success": 0, "physical_success_ever": 0},
        )
        task_row["total"] += 1
        task_row["final_physical_success"] += int(bool(row["final_physical_success"]))
        task_row["physical_success_ever"] += int(bool(row["physical_success_ever"]))

    error_keys = ["mae", "joint_mae", "gripper_abs", "max_abs"]
    mean_error = {}
    for key in error_keys:
        values = [
            row["prediction_error"][key]
            for row in results
            if row.get("prediction_error") is not None and key in row["prediction_error"]
        ]
        if values:
            mean_error[key] = float(np.mean(values))

    return {
        "label": args.label,
        "mode": args.mode,
        "repo_id": args.repo_id,
        "dataset_root": str(args.dataset_root),
        "policy_path": str(args.policy_path) if args.policy_path is not None else None,
        "episodes": [int(row["episode"]) for row in results],
        "total": len(results),
        "final_physical_success": final_ok,
        "physical_success_ever": ever_ok,
        "final_legacy_success": legacy_final_ok,
        "final_physical_success_text": f"{final_ok}/{len(results)}",
        "physical_success_ever_text": f"{ever_ok}/{len(results)}",
        "task_summary": task_summary,
        "mean_prediction_error": mean_error,
        "postprocess": results[0]["postprocess"] if results else None,
    }


def main() -> int:
    args = parse_args()
    prepare_runtime(args)

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
    from mujoco_env.y_env2 import SimpleEnv2

    if args.headless_no_viewer:
        SimpleEnv2.init_viewer = lambda self: self.env.reset()

    dataset = LeRobotDataset(args.repo_id, root=args.dataset_root)
    episode_column = single.tensor_to_np(dataset.hf_dataset["episode_index"]).astype(int)
    available = sorted({int(x) for x in episode_column.tolist()})
    episodes = parse_episode_spec(args.episodes, available)

    policy = None
    if args.mode == "pi0":
        if args.policy_path is None:
            raise ValueError("--policy-path is required for --mode pi0")
        stats_repo_id = args.stats_repo_id or args.repo_id
        stats_dataset_root = args.stats_dataset_root or args.dataset_root
        metadata = LeRobotDatasetMetadata(stats_repo_id, root=stats_dataset_root)
        policy = PI0Policy.from_pretrained(args.policy_path, dataset_stats=metadata.stats)
        policy.to(args.device)
        policy.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with args.output_jsonl.open("w", encoding="utf-8") as jsonl:
        for episode in episodes:
            result = replay_episode(args, dataset, episode_column, episode, policy)
            results.append(result)
            episode_json = args.output_dir / f"ep{int(episode):02d}_{args.mode}.json"
            episode_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            jsonl.write(json.dumps(result, ensure_ascii=False) + "\n")
            jsonl.flush()
            print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = summarize(args, results)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

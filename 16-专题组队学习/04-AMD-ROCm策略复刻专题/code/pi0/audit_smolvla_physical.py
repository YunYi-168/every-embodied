#!/usr/bin/env python3
"""Run SmolVLA rollouts with physical-success auditing.

The project eval script originally reported SmolVLA's environment success only.
This helper keeps that metric but also requires target-mug lift history and final
uprightness, matching the stricter ACT postmortem convention.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from eval_policy_success import (
    get_body_upright_cos,
    get_smolvla_debug,
    make_pi0_policy,
    make_smolvla_policy,
    to_tensor_image,
)


def round_list(values, ndigits: int = 4) -> list[float]:
    return [round(float(x), ndigits) for x in np.asarray(values).reshape(-1)]


def init_tracker(env) -> dict:
    target_body = env.obj_target
    p_target = env.env.get_p_body(target_body)
    p_plate = env.env.get_p_body("body_obj_plate_11")
    return {
        "target_body": target_body,
        "initial_target_pos": round_list(p_target),
        "initial_plate_pos": round_list(p_plate),
        "initial_target_z": float(p_target[2]),
        "max_target_z": float(p_target[2]),
        "max_target_lift": 0.0,
        "lifted_steps": 0,
    }


def update_tracker(env, tracker: dict, args: argparse.Namespace) -> None:
    p_target = env.env.get_p_body(tracker["target_body"])
    target_z = float(p_target[2])
    tracker["max_target_z"] = max(float(tracker["max_target_z"]), target_z)
    lift = target_z - float(tracker["initial_target_z"])
    tracker["max_target_lift"] = max(float(tracker["max_target_lift"]), lift)
    if lift >= float(args.physical_min_lift):
        tracker["lifted_steps"] += 1


def physical_debug(env, tracker: dict, args: argparse.Namespace) -> dict:
    base = get_smolvla_debug(env)
    target_body = tracker["target_body"]
    p_target = env.env.get_p_body(target_body)
    upright_cos = get_body_upright_cos(env, target_body)
    lifted_enough = int(tracker["lifted_steps"]) >= int(args.physical_min_lift_steps)
    final_upright = bool(np.isfinite(upright_cos) and upright_cos >= float(args.physical_final_upright_cos))
    physical_success = bool(base["success"] and lifted_enough and final_upright)
    base.update(
        {
            "physical_success": physical_success,
            "physical_lifted_enough": bool(lifted_enough),
            "physical_final_upright": bool(final_upright),
            "physical_min_lift": float(args.physical_min_lift),
            "physical_min_lift_steps": int(args.physical_min_lift_steps),
            "physical_final_upright_cos_threshold": float(args.physical_final_upright_cos),
            "target_body": target_body,
            "initial_target_pos": tracker["initial_target_pos"],
            "initial_plate_pos": tracker["initial_plate_pos"],
            "final_target_pos": round_list(p_target),
            "max_target_z": float(tracker["max_target_z"]),
            "max_target_lift": float(tracker["max_target_lift"]),
            "lifted_steps": int(tracker["lifted_steps"]),
            "final_target_upright_cos": upright_cos,
        }
    )
    return base


def close_env(env) -> None:
    inner = getattr(env, "env", None)
    if inner is not None:
        try:
            inner.close_viewer()
        except Exception:
            pass
    gc.collect()


def load_action_bounds(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray] | None:
    if not args.clamp_action_to_dataset:
        return None
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    metadata = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)
    action_stats = metadata.stats["action"]
    low = np.asarray(action_stats["min"], dtype=np.float32).reshape(-1)[:7]
    high = np.asarray(action_stats["max"], dtype=np.float32).reshape(-1)[:7]
    return low, high


def postprocess_action(
    action: np.ndarray,
    args: argparse.Namespace,
    action_bounds: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, dict]:
    raw = np.asarray(action, dtype=np.float32).reshape(-1)[:7].copy()
    processed = raw.copy()
    info: dict = {}
    if action_bounds is not None:
        low, high = action_bounds
        processed = np.clip(processed, low, high)
        info["clamped"] = True
    if args.binarize_gripper:
        processed[6] = 1.0 if processed[6] >= float(args.gripper_threshold) else 0.0
        info["binarize_gripper"] = True
        info["gripper_threshold"] = float(args.gripper_threshold)
    if args.gripper_open_until_step is not None and args.current_action_step < int(args.gripper_open_until_step):
        processed[6] = 0.0
        info["gripper_open_until_step"] = int(args.gripper_open_until_step)
    info["raw_action"] = round_list(raw, 5)
    info["processed_action"] = round_list(processed, 5)
    return processed, info


def rollout(args: argparse.Namespace, policy, seed: int, env=None) -> dict:
    from mujoco_env.y_env2 import SimpleEnv2

    owns_env = env is None
    if env is None:
        env = SimpleEnv2("./asset/example_scene_y2.xml", action_type="joint_angle")
    try:
        env.reset(seed=seed)
        if args.instruction:
            env.set_instruction(args.instruction)
        policy.reset()
        tracker = init_tracker(env)
        action_steps = 0
        sim_steps = 0
        success_ever = False
        physical_success_ever = False
        first_success_step = None
        first_physical_success_step = None
        last_debug = physical_debug(env, tracker, args)
        start = time.time()
        action_bounds = load_action_bounds(args)

        while action_steps < args.max_action_steps and env.env.is_viewer_alive():
            env.step_env()
            sim_steps += 1
            if not env.env.loop_every(HZ=args.hz):
                continue

            update_tracker(env, tracker, args)
            last_debug = physical_debug(env, tracker, args)
            if last_debug["success"]:
                success_ever = True
                if first_success_step is None:
                    first_success_step = action_steps
            if last_debug["physical_success"]:
                physical_success_ever = True
                if first_physical_success_step is None:
                    first_physical_success_step = action_steps
            if last_debug["physical_success"]:
                break

            state = env.get_joint_state()[:6]
            image, wrist_image = env.grab_image()
            batch = {
                "observation.state": torch.tensor(np.asarray([state]), dtype=torch.float32, device=args.device),
                "observation.image": to_tensor_image(image).unsqueeze(0).to(args.device),
                "observation.wrist_image": to_tensor_image(wrist_image).unsqueeze(0).to(args.device),
                "task": [env.instruction],
            }
            if args.reset_policy_each_action:
                policy.reset()
            with torch.no_grad():
                action = policy.select_action(batch)[0, :7].detach().cpu().numpy()
            args.current_action_step = action_steps
            action, action_info = postprocess_action(action, args, action_bounds)

            if args.log_steps_jsonl and (action_steps < 10 or action_steps % args.log_every == 0):
                row = {
                    "event": "step",
                    "seed": seed,
                    "action_step": action_steps,
                    "instruction": env.instruction,
                    "action": round_list(action, 5),
                    "action_info": action_info,
                    "debug": last_debug,
                }
                with args.log_steps_jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            env.step(action)
            action_steps += 1
            if args.render:
                env.render()

            update_tracker(env, tracker, args)
            last_debug = physical_debug(env, tracker, args)
            if last_debug["success"]:
                success_ever = True
                if first_success_step is None:
                    first_success_step = action_steps
            if last_debug["physical_success"]:
                physical_success_ever = True
                if first_physical_success_step is None:
                    first_physical_success_step = action_steps
            if last_debug["physical_success"]:
                break

        last_debug = physical_debug(env, tracker, args)
        success = bool(success_ever or last_debug["success"])
        physical_success = bool(physical_success_ever or last_debug["physical_success"])
        last_debug.update(
            {
                "legacy_success_ever": success,
                "first_legacy_success_step": first_success_step,
                "physical_success_ever": physical_success,
                "first_physical_success_step": first_physical_success_step,
            }
        )
        return {
            "policy": args.policy_type,
            "seed": seed,
            "success": success,
            "physical_success": physical_success,
            "action_steps": action_steps,
            "sim_steps": sim_steps,
            "elapsed_s": round(time.time() - start, 3),
            "instruction": getattr(env, "instruction", None),
            "debug": last_debug,
        }
    finally:
        if owns_env:
            close_env(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-type", choices=["smolvla", "pi0"], default="smolvla")
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--max-action-steps", type=int, default=300)
    parser.add_argument("--physical-min-lift", type=float, default=0.03)
    parser.add_argument("--physical-min-lift-steps", type=int, default=3)
    parser.add_argument("--physical-final-upright-cos", type=float, default=0.7)
    parser.add_argument("--reset-policy-each-action", action="store_true")
    parser.add_argument("--clamp-action-to-dataset", action="store_true")
    parser.add_argument("--dataset-repo-id", default="datawhale_eai_pnp_language")
    parser.add_argument("--dataset-root", type=Path, default=Path("./demo_data_language"))
    parser.add_argument("--binarize-gripper", action="store_true")
    parser.add_argument("--gripper-threshold", type=float, default=0.5)
    parser.add_argument(
        "--gripper-open-until-step",
        type=int,
        default=None,
        help="Diagnostic only: force gripper action to 0 before this action step.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--log-steps-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_jsonl.exists():
        args.output_jsonl.unlink()
    if args.log_steps_jsonl:
        args.log_steps_jsonl.parent.mkdir(parents=True, exist_ok=True)
        if args.log_steps_jsonl.exists():
            args.log_steps_jsonl.unlink()
    seeds = args.seeds if args.seeds else list(range(args.seed_start, args.seed_start + args.episodes))

    from mujoco_env.y_env2 import SimpleEnv2

    if args.policy_type == "pi0":
        policy = make_pi0_policy(args.device, args.policy_path)
    else:
        policy = make_smolvla_policy(args.device, args.policy_path)
    results = []
    env = SimpleEnv2("./asset/example_scene_y2.xml", action_type="joint_angle")
    try:
        for seed in seeds:
            row = rollout(args, policy, seed, env=env)
            results.append(row)
            with args.output_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        close_env(env)

    summary = {
        "policy": args.policy_type,
        "episodes": len(results),
        "success_count": sum(1 for row in results if row["success"]),
        "physical_success_count": sum(1 for row in results if row["physical_success"]),
        "success_rate": sum(1 for row in results if row["success"]) / max(len(results), 1),
        "physical_success_rate": sum(1 for row in results if row["physical_success"]) / max(len(results), 1),
        "seeds": seeds,
        "max_action_steps": args.max_action_steps,
        "physical_min_lift": args.physical_min_lift,
        "physical_min_lift_steps": args.physical_min_lift_steps,
        "physical_final_upright_cos": args.physical_final_upright_cos,
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Evaluate a trained skrl policy on an Isaac Lab task and compute episode-level metrics.

Usage (examples):
  python eval_policy.py --task isaaclab_tasks.direct._bionic_hand.bionic_hand_env:MyEnv-Play \
                        --checkpoint /path/to/checkpoint.pt \
                        --num_envs 64 --episodes 200 --rot_tol_deg 5 --pos_tol_cm 2 \
                        --algorithm PPO --ml_framework torch --headless

Notes:
- Mirrors play.py structure so it runs inside Isaac Sim the same way.
- Writes CSV metrics to the experiment log dir next to the checkpoint.
"""

import argparse
import csv
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from packaging import version

from isaaclab.app import AppLauncher

from isaaclab.utils.dict import print_dict

# ---------------------------
# CLI
# ---------------------------
parser = argparse.ArgumentParser(description="Evaluate a checkpoint of an RL agent from skrl.")
parser.add_argument("--task", type=str, required=True, help="Name of the task (same as in play.py).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--agent", type=str, default=None, help="Override agent cfg entry point.")
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"])
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"])
parser.add_argument("--num_envs", type=int, default=None, help="Vectorized envs to simulate.")
parser.add_argument("--episodes", type=int, default=200, help="Total episodes to evaluate (across all envs).")
parser.add_argument("--seed", type=int, default=123, help="Seed for env/agent.")
parser.add_argument("--rot_tol_deg", type=float, default=5.0, help="Orientation tolerance for success (degrees).")
parser.add_argument("--pos_tol_cm", type=float, default=None, help="Optional position tolerance for success (cm).")
parser.add_argument("--max_steps", type=int, default=1500, help="Max steps per episode before truncate.")
parser.add_argument("--real-time", action="store_true", help="Sleep to match sim step_dt.")
parser.add_argument("--video", action="store_true", help="(Optional) record a short video just for sanity check.")
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
# Append AppLauncher args (e.g., --headless, --renderer, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

# Hydra needs clean argv
sys.argv = [sys.argv[0]] + hydra_args

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import skrl
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.skrl import SkrlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# ---------------------------
# skrl version check & imports
# ---------------------------
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    raise RuntimeError(f"skrl {SKRL_VERSION}+ required, found {skrl.__version__}")

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner


# ---------------------------
# Helpers
# ---------------------------

def to_np(x):
    import numpy as np
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: to_np(v) for k, v in x.items()}
    return np.asarray(x)


def quat_geodesic_deg(q_goal: np.ndarray, q_obj: np.ndarray) -> np.ndarray:
    q_goal = np.asarray(q_goal)
    q_obj = np.asarray(q_obj)
    q_goal = q_goal / (np.linalg.norm(q_goal, axis=-1, keepdims=True) + 1e-8)
    q_obj = q_obj / (np.linalg.norm(q_obj, axis=-1, keepdims=True) + 1e-8)

    def dot(q1, q2):
        return np.sum(q1 * q2, axis=-1)

    d = np.abs(dot(q_goal, q_obj))
    d = np.clip(d, -1.0, 1.0)
    ang = 2.0 * np.arccos(d)
    return np.degrees(ang)


def l2(a):
    return float(np.linalg.norm(a))


def maybe_from_infos(infos):
    """Try to extract pose data from the infos returned by env.step(...).
    Works for both dict-of-tensors and list[dict] structures.
    """
    if infos is None:
        return None
    out = {}
    # Case A: dict of tensors (vectorized)
    if isinstance(infos, dict):
        for k in ("goal_quat", "cube_quat", "goal_pos", "cube_pos", "dropped", "is_success_step"):
            if k in infos:
                out[k] = to_np(infos[k])
    # Case B: list of dicts per env -> stack if keys exist in first dict
    elif isinstance(infos, (list, tuple)) and len(infos) > 0 and isinstance(infos[0], dict):
        keys = ("goal_quat", "cube_quat", "goal_pos", "cube_pos", "dropped", "is_success_step")
        for k in keys:
            if all(k in d for d in infos):
                out[k] = to_np(np.stack([d[k] for d in infos], axis=0))
    return out if out else None


def maybe_from_extras(env):
    try:
        ex = env.unwrapped.extras
    except Exception:
        return None
    out = {}
    for k in ("goal_quat", "cube_quat", "goal_pos", "cube_pos", "dropped", "is_success_step"):
        if k in ex:
            out[k] = to_np(ex[k])
    return out if out else None


def maybe_from_getters(env):
    out = {}
    g = getattr(env.unwrapped, "get_goal_pose", None)
    c = getattr(env.unwrapped, "get_object_pose", None)
    d = getattr(env.unwrapped, "get_dropped_flags", None)
    if g is not None and c is not None:
        gp = g()
        op = c()
        if isinstance(gp, dict):
            out["goal_pos"] = to_np(gp.get("pos"))
            out["goal_quat"] = to_np(gp.get("quat"))
        else:
            out["goal_pos"], out["goal_quat"] = map(to_np, gp)
        if isinstance(op, dict):
            out["cube_pos"] = to_np(op.get("pos"))
            out["cube_quat"] = to_np(op.get("quat"))
        else:
            out["cube_pos"], out["cube_quat"] = map(to_np, op)
    if d is not None:
        out["dropped"] = to_np(d())
    return out if out else None


def compute_metrics(batch_data, rot_tol_deg, pos_tol_m=None):
    goal_q = batch_data.get("goal_quat")
    cube_q = batch_data.get("cube_quat")
    goal_p = batch_data.get("goal_pos")
    cube_p = batch_data.get("cube_pos")
    dropped = batch_data.get("dropped")

    metrics = {}
    rot_err_deg = None
    pos_err_m = None

    if goal_q is not None and cube_q is not None:
        rot_err_deg = quat_geodesic_deg(goal_q, cube_q)
        metrics["rot_err_deg"] = rot_err_deg

    if goal_p is not None and cube_p is not None:
        pos_err_m = np.linalg.norm(goal_p - cube_p, axis=-1)
        metrics["pos_err_m"] = pos_err_m

    if dropped is not None:
        metrics["dropped"] = np.asarray(dropped).astype(np.int32)

    if rot_err_deg is not None:
        success_mask = rot_err_deg <= rot_tol_deg
        if pos_err_m is not None and pos_tol_m is not None:
            success_mask = np.logical_and(success_mask, pos_err_m <= pos_tol_m)
        metrics["is_success"] = success_mask.astype(np.int32)

    return metrics


def make_log_dir_and_csv(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "evaluation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "episode",
            "env_id",
            "return",
            "length",
            "final_rot_err_deg",
            "final_pos_err_cm",
            "success",
            "time_to_success_steps",
            "dropped"
        ])
    return csv_path


# --------------------------------------
# Hydra config hook, env + agent creation
# --------------------------------------
if args_cli.agent is None:
    algo = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algo in ["ppo"] else f"skrl_{algo}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent

# Removed unsupported import of get_published_pretrained_checkpoint

@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{args_cli.algorithm.lower()}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    experiment_cfg["seed"] = args_cli.seed
    env_cfg.seed = args_cli.seed

    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    
    # Pre-trained checkpoint logic
    if args_cli.use_pretrained_checkpoint:
        try:
            # Note: This function is not available in the provided code, but is used in the original Isaac Lab.
            # Assuming it's a utility that gets the path to a published checkpoint.
            from isaaclab_tasks.utils import get_published_pretrained_checkpoint
            resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
            if not resume_path:
                print("[INFO] No pre-trained checkpoint available for this task.")
                return
        except ImportError:
            print("[WARN] Could not import 'get_published_pretrained_checkpoint'. Skipping pre-trained checkpoint.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{args_cli.algorithm.lower()}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    import gymnasium as gym
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=("rgb_array" if args_cli.video else None))
    if isinstance(env.unwrapped, DirectMARLEnv) and args_cli.algorithm.lower() in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "eval"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording a short video (sanity check).")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    runner = Runner(env, experiment_cfg)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.set_running_mode("eval")

    E = env.num_envs
    total_eps_target = args_cli.episodes
    episode_returns = np.zeros(E, dtype=np.float32)
    episode_lengths = np.zeros(E, dtype=np.int32)
    first_success_step = np.full(E, -1, dtype=np.int32)

    csv_path = make_log_dir_and_csv(log_dir)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)

    rot_tol_deg = args_cli.rot_tol_deg
    pos_tol_m = (args_cli.pos_tol_cm / 100.0) if args_cli.pos_tol_cm is not None else None

    obs, _ = env.reset()
    finished_episodes = 0
    timestep = 0

    print("[INFO] Starting evaluation...")
    warned_missing_metrics = False
    while simulation_app.is_running() and finished_episodes < total_eps_target:
        t0 = time.time()
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            obs, rewards, terminated, truncated, infos = env.step(actions)

        r_np = to_np(rewards).astype(np.float32).reshape(-1)
        episode_returns += r_np
        episode_lengths += 1

        batch = maybe_from_infos(infos) or maybe_from_extras(env) or maybe_from_getters(env) or {}
        metrics = compute_metrics(batch, rot_tol_deg, pos_tol_m)

        if "is_success" in metrics:
            succ = metrics["is_success"].astype(bool)
            newly = (first_success_step < 0) & succ
            first_success_step[newly] = episode_lengths[newly]
        else:
            if not warned_missing_metrics:
                print(
                "[WARN] No pose data found in infos/extras/getters; final_rot_err_deg will be NaN. "
                "Add per-step in your env: extras['goal_quat'], extras['cube_quat'], "
                "extras['goal_pos'], extras['cube_pos'] (torch tensors ok)."
                )
                warned_missing_metrics = True

        term_np = to_np(terminated)
        trunc_np = to_np(truncated)
        done = np.logical_or(term_np, trunc_np).astype(bool).reshape(-1)
        done_idx = np.where(done)[0]
        for i in done_idx:
            ep_ret = float(episode_returns[i])
            ep_len = int(episode_lengths[i])

            final_rot = float(metrics.get("rot_err_deg", np.array([np.nan] * E))[i])
            final_pos_cm = (
                float(metrics.get("pos_err_m", np.array([np.nan] * E))[i] * 100.0) if "pos_err_m" in metrics else np.nan
            )
            dropped = int(metrics.get("dropped", np.zeros(E))[i]) if "dropped" in metrics else 0

            success_flag = 1 if first_success_step[i] > 0 else 0
            tts = int(first_success_step[i]) if success_flag == 1 else -1

            writer.writerow([finished_episodes + 1, i, ep_ret, ep_len, final_rot, final_pos_cm, success_flag, tts, dropped])

            episode_returns[i] = 0.0
            episode_lengths[i] = 0
            first_success_step[i] = -1

            finished_episodes += 1
            if finished_episodes >= total_eps_target:
                break

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        if args_cli.real_time:
            sleep_time = dt - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    csv_file.close()
    env.close()

    import pandas as pd
    df = pd.read_csv(csv_path)
    sr = df["success"].mean() * 100.0 if "success" in df else float("nan")
    dr = df["dropped"].mean() * 100.0 if "dropped" in df else float("nan")
    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {len(df)}")
    print(f"Success rate: {sr:.1f}%")
    if "time_to_success_steps" in df:
        tts = df.loc[df["time_to_success_steps"] > 0, "time_to_success_steps"]
        if len(tts):
            print(f"Median time-to-success (steps): {tts.median():.1f}")
    if "final_rot_err_deg" in df:
        print(f"Median final rot error (deg): {df['final_rot_err_deg'].median():.2f}")
    if "final_pos_err_cm" in df:
        print(f"Median final pos error (cm): {df['final_pos_err_cm'].median():.2f}")
    if "dropped" in df:
        print(f"Drop rate: {dr:.1f}%")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()

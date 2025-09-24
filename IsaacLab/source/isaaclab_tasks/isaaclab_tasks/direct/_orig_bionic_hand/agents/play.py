# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Evaluate a trained skrl policy on an Isaac Lab task and compute episode-level metrics.

Usage (examples):
  ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/_orig_bionic_hand/agents/play.py \
    --task Isaac-Repose-Cube-OrigBionicHand-Direct-v0 \
    --algorithm PPO \
    --checkpoint /workspace/isaaclab/logs/skrl/bionic_hand/2025-09-23_01-50-23_ppo_torch/checkpoints/best_agent.pt \
    --num_envs 9 --episodes 200 --rot_tol_deg 5 --headless

Notes:
- Mirrors the usual play flow (Hydra + Skrl Runner) so it runs inside Isaac Sim.
- Writes evaluation CSV next to the checkpoint: <.../logs/skrl/.../evaluation.csv>
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
    # enable camera / recorder specific flags if your task supports it
    pass

# Hydra needs clean argv
sys.argv = [sys.argv[0]] + hydra_args

# Launch app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------
# skrl version check & imports
# ---------------------------
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

SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    raise RuntimeError(f"skrl {SKRL_VERSION}+ required, found {skrl.__version__}")

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
else:
    raise NotImplementedError("This evaluation helper currently targets PyTorch skrl only.")

# ---------------------------
# Helpers
# ---------------------------
def to_np(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def quat_geodesic_deg(q1, q2):
    # q1,q2: (E,4) wxyz (IsaacLab convention)
    # angle = 2*acos(|<q1,q2>|)
    dot = np.sum(q1 * q2, axis=-1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dot))

def maybe_from_infos(infos):
    # infos is list/tuple/dict coming from vectorized env
    # try to gather batch tensors from info / extras
    batch = defaultdict(list)
    if isinstance(infos, (list, tuple)):
        for i in infos:
            if isinstance(i, dict):
                e = i.get("extras", {})
                for k, v in e.items():
                    batch[k].append(v)
    elif isinstance(infos, dict):
        e = infos.get("extras", {})
        for k, v in e.items():
            batch[k] = v
    # stack
    out = {}
    for k, v in batch.items():
        try:
            out[k] = to_np(torch.stack(v)) if isinstance(v, list) and isinstance(v[0], torch.Tensor) else to_np(v)
        except Exception:
            out[k] = to_np(v)
    return out if out else None

def maybe_from_extras(env):
    try:
        ex = getattr(env.unwrapped, "extras", None)
        if not ex:
            return None
        out = {}
        for k in ("goal_pos", "goal_quat", "cube_pos", "cube_quat",
                  "dropped", "is_success_step", "num_goal_hits"):
            if k in ex:
                out[k] = to_np(ex[k])
        return out if out else None
    except Exception:
        return None

def maybe_from_getters(env):
    out = {}
    g = getattr(env.unwrapped, "get_goal_pose", None)
    c = getattr(env.unwrapped, "get_object_pose", None)
    d = getattr(env.unwrapped, "get_dropped_flags", None)
    s = getattr(env.unwrapped, "get_is_success_step", None)

    # (num_goal_hits) — only if you’ve actually added a getter or expose `successes`
    get_hits = getattr(env.unwrapped, "get_num_goal_hits", None)
    successes_attr = getattr(env.unwrapped, "successes", None)

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
    if s is not None:
        out["is_success_step"] = to_np(s())
    if get_hits is not None:
        out["num_goal_hits"] = to_np(get_hits())
    elif isinstance(successes_attr, torch.Tensor):
        out["num_goal_hits"] = to_np(successes_attr).astype(np.int32)

    return out if out else None


def compute_metrics(batch_data, rot_tol_deg, pos_tol_m=None):
    goal_q = batch_data.get("goal_quat")
    cube_q = batch_data.get("cube_quat")
    goal_p = batch_data.get("goal_pos")
    cube_p = batch_data.get("cube_pos")

    metrics = {}
    if goal_q is not None and cube_q is not None:
        dot = np.sum(goal_q * cube_q, axis=-1)
        dot = np.clip(np.abs(dot), -1.0, 1.0)
        metrics["rot_err_deg"] = np.rad2deg(2.0 * np.arccos(dot))

    if goal_p is not None and cube_p is not None:
        metrics["pos_err_m"] = np.linalg.norm(goal_p - cube_p, axis=-1)

    if "dropped" in batch_data and batch_data["dropped"] is not None:
        metrics["dropped"] = np.asarray(batch_data["dropped"]).astype(np.int32)

    if "num_goal_hits" in batch_data and batch_data["num_goal_hits"] is not None:
        metrics["num_goal_hits"] = np.asarray(batch_data["num_goal_hits"]).astype(np.int32)

    # Prefer env-provided per-step success
    if "is_success_step" in batch_data and batch_data["is_success_step"] is not None:
        metrics["is_success"] = np.asarray(batch_data["is_success_step"]).astype(np.int32)
    elif "rot_err_deg" in metrics:
        success_mask = metrics["rot_err_deg"] <= rot_tol_deg
        if "pos_err_m" in metrics and pos_tol_m is not None:
            success_mask = np.logical_and(success_mask, metrics["pos_err_m"] <= pos_tol_m)
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
            "min_rot_err_deg",
            "final_pos_err_cm",
            "success",                 # 1 if success ever happened within episode
            "time_to_success_steps",   # first success step or -1
            "num_goal_hits",           # NEW: total successes within episode
            "dropped"                  # 1 if episode ended by drop
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


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    algorithm = args_cli.algorithm.lower()
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0  # don't generate checkpoints
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")
    E = env.num_envs
    total_eps_target = args_cli.episodes
    episode_returns = np.zeros(E, dtype=np.float32)
    episode_lengths = np.zeros(E, dtype=np.int32)
    finished_episodes = 0
    min_rot_err = np.full(E, np.inf, dtype=np.float32)
    first_success_step = np.full(E, -1, dtype=np.int32)
    last_num_goal_hits = np.zeros(E, dtype=np.int32)
    success_ever_total = 0
    dropped_total = 0
    hits_sum = 0
    rot_tol_deg = args_cli.rot_tol_deg
    pos_tol_m = (args_cli.pos_tol_cm / 100.0) if args_cli.pos_tol_cm is not None else None

    csv_path = make_log_dir_and_csv(log_dir)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)
    # reset environment
    obs, _ = env.reset()
    timestep = 0
    total_successes = 0
    total_drops = 0
    total_returns = 0.0
    total_lengths = 0
    
    print("[INFO] Starting evaluation...")
    warned_missing_metrics = False
    while simulation_app.is_running() and finished_episodes < total_eps_target:
        start_time = time.time()
        with torch.inference_mode():
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, rewards, terminated, truncated, infos = env.step(actions)

        r_np = to_np(rewards).astype(np.float32).reshape(-1)
        episode_returns += r_np
        episode_lengths += 1

        # Batch metrics from infos/extras (or getters as fallback)
        batch = maybe_from_infos(infos) or maybe_from_extras(env) or maybe_from_getters(env) or {}
        metrics = compute_metrics(batch, rot_tol_deg, pos_tol_m)
        if "rot_err_deg" in metrics:
            min_rot_err = np.minimum(min_rot_err, metrics["rot_err_deg"])

        # Compute first success step
        if "is_success" in metrics:
            newly = (first_success_step < 0) & metrics["is_success"].astype(bool)
            first_success_step[newly] = episode_lengths[newly]
        
        if "num_goal_hits" in metrics:
            # metrics["num_goal_hits"] is per-env count at *this* step; keep the latest
            last_num_goal_hits = metrics["num_goal_hits"].astype(np.int32)


        # Check episode ends per env
        term = to_np(terminated).reshape(-1)
        trunc = to_np(truncated).reshape(-1)
        done_mask = np.logical_or(term, trunc).astype(bool)

        # On episode finished: write a row
        for i, done in enumerate(done_mask):
            if not done:
                continue
            ep_ret = float(episode_returns[i])
            ep_len = int(episode_lengths[i])

            final_rot = float(metrics.get("rot_err_deg", np.array([np.nan] * E))[i])
            final_pos_cm = (
                float(metrics.get("pos_err_m", np.array([np.nan] * E))[i] * 100.0)
                if "pos_err_m" in metrics else np.nan
            )
            dropped = int(metrics.get("dropped", np.zeros(E))[i]) if "dropped" in metrics else 0

            success_flag = 1 if first_success_step[i] > 0 else 0
            tts = int(first_success_step[i]) if success_flag == 1 else -1
            hits = int(last_num_goal_hits[i])
            
            writer.writerow([finished_episodes + 1, i, ep_ret, ep_len,
                            final_rot, float(min_rot_err[i]), final_pos_cm,
                            success_flag, tts, int(last_num_goal_hits[i]),dropped])
            total_successes += success_flag
            total_drops += dropped
            total_returns += ep_ret
            total_lengths += ep_len
            hits_sum += hits

            # reset per-env trackers
            episode_returns[i] = 0.0
            episode_lengths[i] = 0
            first_success_step[i] = -1
            min_rot_err[i] = np.inf

            finished_episodes += 1
            if finished_episodes >= total_eps_target:
                break

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        if args_cli.real_time:
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    csv_file.close()
    csv_file.close()

    num_eps = max(finished_episodes, 1)
    succ_rate = 100.0 * total_successes / num_eps      # % episodes with >=1 success
    drop_rate = 100.0 * total_drops / num_eps          # % episodes ended via drop
    avg_hits = hits_sum / num_eps
    avg_return = total_returns / num_eps
    avg_len = total_lengths / num_eps

    print("\n[INFO] Evaluation complete.")
    print(f"[INFO] Episodes: {finished_episodes}")
    print(f"[INFO] Ever-success rate: {succ_rate:.1f}%  (episodes with ≥1 success / total)")
    print(f"[INFO] Drop rate:         {drop_rate:.1f}%  (episodes ended by drop)")
    print(f"[INFO] Avg successes per episode: {avg_hits:.2f}")
    print(f"[INFO] Avg return: {avg_return:.2f}   Avg length: {avg_len:.1f} steps")
    print(f"[INFO] CSV: {csv_path}")

    env.close()

    

if __name__ == "__main__":
    main()

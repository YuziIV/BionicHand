#!/usr/bin/env python3
"""
Plot Isaac Lab evaluation.csv and visualize "closeness" to success.

Usage:
  python plot_eval.py /path/to/evaluation.csv --rot_tol_deg 5 --near_tol_deg 15 --dpi 140

Notes:
- "Near miss" is defined as: success==0 AND dropped==0 AND min_rot_err_deg <= near_tol_deg.
- If you also care about position, add --near_pos_cm to require final_pos_err_cm <= that threshold for a near miss.
"""

import argparse
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", type=str, help="Path to evaluation.csv")
    p.add_argument("--rot_tol_deg", type=float, default=5.0, help="Success orientation tolerance (deg)")
    p.add_argument("--near_tol_deg", type=float, default=15.0, help="Near-miss orientation threshold (deg)")
    p.add_argument("--near_pos_cm", type=float, default=None,
                   help="Optional: also require final_pos_err_cm <= this for a near miss")
    p.add_argument("--dpi", type=int, default=140, help="Figure DPI")
    p.add_argument("--max_episodes", type=int, default=None, help="Optionally limit number of rows (debug)")
    return p.parse_args()

def as_num(s):
    return pd.to_numeric(s, errors="coerce")

def load_csv(path, max_rows=None):
    df = pd.read_csv(path)
    if max_rows is not None:
        df = df.head(max_rows)
    # Coerce numeric columns safely
    for col in [
        "episode", "env_id", "return", "length",
        "final_rot_err_deg", "min_rot_err_deg",
        "final_pos_err_cm", "success", "time_to_success_steps",
        "num_goal_hits", "dropped"
    ]:
        if col in df.columns:
            df[col] = as_num(df[col])
    return df

def categorize(df, rot_tol_deg, near_tol_deg, near_pos_cm=None):
    df = df.copy()
    df["success"] = df["success"].fillna(0).astype(int)
    df["dropped"] = df["dropped"].fillna(0).astype(int)

    # Gaps (negative or ~0 => solved/at tol)
    df["rot_gap_deg"] = df["min_rot_err_deg"] - rot_tol_deg

    # Category labels
    solved = df["success"] == 1
    dropped = df["dropped"] == 1
    nosolve_nodrop = (~solved) & (~dropped)

    near = nosolve_nodrop & (df["min_rot_err_deg"] <= near_tol_deg)
    if (near_pos_cm is not None) and ("final_pos_err_cm" in df.columns):
        near = near & (df["final_pos_err_cm"] <= near_pos_cm)

    both = solved & dropped  # solved at least once, then eventually dropped

    df["category"] = "no-success/no-drop"
    df.loc[dropped & ~solved, "category"] = "drop"
    df.loc[near, "category"] = "near-miss"
    df.loc[solved & ~dropped, "category"] = "success"
    df.loc[both, "category"] = "success+drop"

    return df

def ensure_out_dir(csv_path):
    root = os.path.dirname(os.path.abspath(csv_path))
    out_dir = os.path.join(root, "plots")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def savefig(out_dir, name, dpi):
    path = os.path.join(out_dir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[saved] {path}")

def plot_category_counts(df, out_dir, dpi):
    counts = df["category"].value_counts().reindex(
        ["success", "success+drop", "near-miss", "drop", "no-success/no-drop"], fill_value=0
    )
    total = max(len(df), 1)
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(counts.index, counts.values)
    plt.ylabel("Episodes")
    plt.title("Episode outcome counts")

    for bar, val in zip(bars, counts.values):
        pct = 100.0 * val / total
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f"{val}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=9)
    savefig(out_dir, "01_counts.png", dpi)
    plt.close()

def plot_min_rot_by_episode(df, rot_tol_deg, near_tol_deg, out_dir, dpi):
    # Color-coded scatter of min_rot_err_deg vs episode index
    order = df.sort_values("episode")
    x = order["episode"].values
    y = order["min_rot_err_deg"].values

    plt.figure(figsize=(12, 5))
    masks = {
        "success": order["category"] == "success",
        "success+drop": order["category"] == "success+drop",
        "near-miss": order["category"] == "near-miss",
        "drop": order["category"] == "drop",
        "no-success/no-drop": order["category"] == "no-success/no-drop",
    }
    styles = {
        "success": dict(marker="o", s=30, alpha=0.9),
        "success+drop": dict(marker="P", s=40, alpha=0.9),
        "near-miss": dict(marker="^", s=30, alpha=0.9),
        "drop": dict(marker="x", s=40, alpha=0.9),
        "no-success/no-drop": dict(marker=".", s=20, alpha=0.5),
    }
    colors = {
        "success": "tab:green",
        "success+drop": "tab:olive",
        "near-miss": "tab:orange",
        "drop": "tab:red",
        "no-success/no-drop": "tab:gray",
    }

    for cat, m in masks.items():
        if m.any():
            plt.scatter(x[m], y[m], label=cat, c=colors[cat], **styles[cat])

    plt.axhline(rot_tol_deg, ls="--", lw=1, label=f"rot_tol={rot_tol_deg}°")
    plt.axhline(near_tol_deg, ls=":", lw=1, label=f"near_tol={near_tol_deg}°")
    plt.xlabel("Episode")
    plt.ylabel("Min rotation error (deg)")
    plt.title("Min rotation error per episode")
    plt.legend(ncol=3, fontsize=9)
    savefig(out_dir, "02_min_rot_by_episode.png", dpi)
    plt.close()

def plot_min_rot_hist(df, rot_tol_deg, near_tol_deg, out_dir, dpi):
    plt.figure(figsize=(8, 4.5))
    vals = df["min_rot_err_deg"].dropna().values
    bins = min(50, max(10, int(math.sqrt(len(vals))) if len(vals) > 0 else 10))
    plt.hist(vals, bins=bins, alpha=0.85)
    plt.axvline(rot_tol_deg, ls="--", lw=1, label=f"rot_tol={rot_tol_deg}°")
    plt.axvline(near_tol_deg, ls=":", lw=1, label=f"near_tol={near_tol_deg}°")
    plt.xlabel("Min rotation error (deg)")
    plt.ylabel("Episodes")
    plt.title("Distribution: min rotation error per episode")
    plt.legend()
    savefig(out_dir, "03_min_rot_hist.png", dpi)
    plt.close()

def plot_hits_and_len(df, out_dir, dpi):
    fig, ax1 = plt.subplots(figsize=(12, 4.5))
    order = df.sort_values("episode")
    ep = order["episode"].values
    hits = order["num_goal_hits"].fillna(0).values
    ln = order["length"].values

    ax1.plot(ep, hits, label="num_goal_hits", marker="o", ms=3, lw=1)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Hits per episode")
    ax1.grid(alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(ep, ln, label="length", marker=".", ms=3, lw=1, alpha=0.6, color="tab:orange")
    ax2.set_ylabel("Length (steps)")

    lines = [
        Line2D([0], [0], color="tab:blue", lw=2, label="num_goal_hits"),
        Line2D([0], [0], color="tab:orange", lw=2, label="length"),
    ]
    ax1.legend(handles=lines, loc="upper right")
    plt.title("Goal hits and episode length")
    savefig(out_dir, "04_hits_and_length.png", dpi)
    plt.close()

def plot_tts_hist(df, out_dir, dpi):
    suc = df[df["success"] == 1]
    if len(suc) == 0:
        return
    tts = suc["time_to_success_steps"].dropna()
    plt.figure(figsize=(8, 4.5))
    bins = min(50, max(10, int(math.sqrt(len(tts))) if len(tts) > 0 else 10))
    plt.hist(tts, bins=bins, alpha=0.85)
    plt.xlabel("Time to first success (steps)")
    plt.ylabel("Episodes")
    plt.title("Distribution: time to first success (successful episodes only)")
    savefig(out_dir, "05_tts_hist.png", dpi)
    plt.close()

def plot_rot_gap(df, out_dir, dpi):
    order = df.sort_values("episode")
    ep = order["episode"].values
    gap = order["rot_gap_deg"].values
    plt.figure(figsize=(12, 4.5))
    plt.plot(ep, gap, marker=".", lw=1, alpha=0.9)
    plt.axhline(0.0, ls="--", lw=1, color="k", label="0 = at tolerance")
    plt.xlabel("Episode")
    plt.ylabel("min_rot_err_deg - rot_tol_deg (deg)")
    plt.title("Closeness to orientation tolerance (negative = within tol)")
    plt.legend()
    savefig(out_dir, "06_rot_gap_by_episode.png", dpi)
    plt.close()

def per_env_hits_bar(df, out_dir, dpi):
    if "env_id" not in df.columns:
        return
    by_env = df.groupby("env_id")["num_goal_hits"].sum().sort_index()
    plt.figure(figsize=(10, 4))
    plt.bar(by_env.index.astype(int), by_env.values)
    plt.xlabel("env_id")
    plt.ylabel("Total goal hits")
    plt.title("Goal hits per environment (across episodes)")
    savefig(out_dir, "07_hits_per_env.png", dpi)
    plt.close()

def print_summary(df, rot_tol_deg, near_tol_deg, near_pos_cm):
    N = len(df)
    succ = int((df["success"] == 1).sum())
    drop = int((df["dropped"] == 1).sum())
    succ_rate = 100.0 * succ / max(N, 1)
    drop_rate = 100.0 * drop / max(N, 1)
    avg_hits = df["num_goal_hits"].fillna(0).mean()
    avg_ret = df["return"].mean()
    avg_len = df["length"].mean()

    near_miss = (df["category"] == "near-miss").sum()
    near_rate = 100.0 * near_miss / max(N, 1)

    print("\n=== SUMMARY ===")
    print(f"Episodes: {N}")
    print(f"Success rate: {succ_rate:.1f}%    (episodes with ≥1 success)")
    print(f"Drop rate:    {drop_rate:.1f}%    (episodes ended via drop)")
    print(f"Near-miss:    {near_miss} ({near_rate:.1f}%)  "
          f"[criteria: min_rot_err <= {near_tol_deg}°"
          + (f" & final_pos_err <= {near_pos_cm} cm" if near_pos_cm is not None else "") + "]")
    print(f"Avg successes/episode: {avg_hits:.2f}")
    print(f"Avg return: {avg_ret:.2f}   Avg length: {avg_len:.1f} steps")
    print("===============")

    # Top 20 closest non-success episodes
    nosolved = df[(df["success"] == 0)]
    top = nosolved.sort_values("rot_gap_deg").head(20)[
        ["episode", "env_id", "min_rot_err_deg", "rot_gap_deg", "dropped", "num_goal_hits", "length"]
    ]
    if len(top) > 0:
        print("\nTop near-misses by rotation gap (best 20 among non-success):")
        print(top.to_string(index=False))

def save_near_miss_table(df, out_dir):
    nm = df[df["category"] == "near-miss"].copy()
    if len(nm) == 0:
        return
    keep = ["episode", "env_id", "min_rot_err_deg", "rot_gap_deg", "final_pos_err_cm", "length"]
    keep = [c for c in keep if c in nm.columns]
    out_csv = os.path.join(out_dir, "near_misses.csv")
    nm[keep].to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

def main():
    args = parse_args()
    df = load_csv(args.csv_path, args.max_episodes)
    out_dir = ensure_out_dir(args.csv_path)
    df = categorize(df, args.rot_tol_deg, args.near_tol_deg, args.near_pos_cm)

    print_summary(df, args.rot_tol_deg, args.near_tol_deg, args.near_pos_cm)
    save_near_miss_table(df, out_dir)

    plot_category_counts(df, out_dir, args.dpi)
    plot_min_rot_by_episode(df, args.rot_tol_deg, args.near_tol_deg, out_dir, args.dpi)
    plot_min_rot_hist(df, args.rot_tol_deg, args.near_tol_deg, out_dir, args.dpi)
    plot_hits_and_len(df, out_dir, args.dpi)
    plot_tts_hist(df, out_dir, args.dpi)
    plot_rot_gap(df, out_dir, args.dpi)
    per_env_hits_bar(df, out_dir, args.dpi)

    print(f"\nDone. Plots written to: {out_dir}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def outcome_masks(df, succ_deg, near_hi):
    succ = (df["success"] == 1)
    drop = (df["dropped"] == 1)
    mred = df["min_rot_err_deg"].astype(float)
    near = (~succ) & (~drop) & mred.notna() & (mred > succ_deg) & (mred <= near_hi)
    stuck = (~succ) & (~drop) & (~near)
    return succ, near, drop, stuck

def add_tol_lines(ax, succ_deg, near_hi, label_prefix=""):
    ax.axvline(succ_deg, linestyle="--", linewidth=2, label=f"{label_prefix}success tol = {succ_deg:.2f}°")
    ax.axvspan(succ_deg, near_hi, alpha=0.2, label=f"{label_prefix}near +{near_hi - succ_deg:.1f}°")

def safe_hist(ax, series, bins, title, xlabel):
    s = series.dropna()
    if len(s) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("#episodes")
        return
    ax.hist(s, bins=bins)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("#episodes")

def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def main():
    p = argparse.ArgumentParser(description="Plot IsaacLab eval CSV with tolerance highlighting and near-misses.")
    p.add_argument("csv_path", type=str, help="Path to evaluation.csv")
    p.add_argument("--success_tol_rad", type=float, default=None,
                   help="Env success tolerance in radians (e.g., 0.4). If set, overrides --success_tol_deg.")
    p.add_argument("--success_tol_deg", type=float, default=None,
                   help="Env success tolerance in degrees. Ignored if --success_tol_rad is set.")
    p.add_argument("--near_extra_deg", type=float, default=10.0,
                   help="Near-miss band size above success tolerance (degrees).")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--max_episodes", type=int, default=None,
                   help="Only use first N rows.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Where to save plots/CSVs. Default: <csv_dir>/eval_plots")
    args = p.parse_args()

    df = pd.read_csv(args.csv_path)
    if args.max_episodes is not None:
        df = df.iloc[: args.max_episodes].copy()

    needed = ["episode","env_id","return","length","final_rot_err_deg","min_rot_err_deg",
              "final_pos_err_cm","success","time_to_success_steps","num_goal_hits","dropped"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"CSV missing column: {c}")

    df["success"] = df["success"].fillna(0).astype(int)
    df["dropped"] = df["dropped"].fillna(0).astype(int)
    for col in ["final_rot_err_deg","min_rot_err_deg","final_pos_err_cm","return","length","time_to_success_steps","num_goal_hits"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # success tolerance in deg
    if args.success_tol_rad is not None:
        succ_deg = float(np.degrees(args.success_tol_rad))
    elif args.success_tol_deg is not None:
        succ_deg = float(args.success_tol_deg)
    else:
        succ_deg = float(np.degrees(0.4))  # default matches your env (0.4 rad)
    near_hi = succ_deg + float(args.near_extra_deg)

    # output folder (like before)
    csv_dir = os.path.dirname(os.path.abspath(args.csv_path))
    out_dir = args.out_dir or os.path.join(csv_dir, "eval_plots")
    os.makedirs(out_dir, exist_ok=True)

    # masks
    succ_mask, near_mask, drop_mask, stuck_mask = outcome_masks(df, succ_deg, near_hi)

    # console summary
    n = len(df)
    n_succ = int(succ_mask.sum())
    n_near = int(near_mask.sum())
    n_drop = int(drop_mask.sum())
    n_stuck = int(stuck_mask.sum())
    print("\n[INFO] Evaluation summary")
    print(f"[INFO] Episodes:                   {n}")
    print(f"[INFO] Success tolerance:          {succ_deg:.3f}° ({np.radians(succ_deg):.3f} rad)")
    print(f"[INFO] Near-miss band:             ({succ_deg:.3f}°, {near_hi:.3f}°]")
    print(f"[INFO] Successes:                  {n_succ} ({100.0 * n_succ / n:.1f}%)")
    print(f"[INFO] Near misses (no drop):      {n_near} ({100.0 * n_near / n:.1f}%)")
    print(f"[INFO] Drops:                      {n_drop} ({100.0 * n_drop / n:.1f}%)")
    print(f"[INFO] Stuck (no success, no drop):{n_stuck} ({100.0 * n_stuck / n:.1f}%)")
    print(f"[INFO] Saving outputs to: {out_dir}")

    # CSVs in folder
    outcome = np.where(succ_mask, "success",
               np.where(near_mask, "near-miss",
               np.where(drop_mask, "drop", "stuck")))
    out_ep = df.copy()
    out_ep["outcome"] = outcome
    out_ep.to_csv(os.path.join(out_dir, "episode_outcomes.csv"), index=False)

    df.loc[near_mask, :].sort_values("min_rot_err_deg").to_csv(
        os.path.join(out_dir, "near_misses.csv"), index=False
    )

    # ---- Separate plots ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
    safe_hist(ax, df["min_rot_err_deg"], bins=40, title="Min rotation error per episode (deg)", xlabel="min_rot_err_deg")
    add_tol_lines(ax, succ_deg, near_hi); ax.legend()
    savefig(fig, os.path.join(out_dir, "hist_min_rot_deg.png"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
    safe_hist(ax, df["final_rot_err_deg"], bins=40, title="Final rotation error per episode (deg)", xlabel="final_rot_err_deg")
    add_tol_lines(ax, succ_deg, near_hi); ax.legend()
    savefig(fig, os.path.join(out_dir, "hist_final_rot_deg.png"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
    safe_hist(ax, df["final_pos_err_cm"], bins=40, title="Final position error per episode (cm)", xlabel="final_pos_err_cm")
    savefig(fig, os.path.join(out_dir, "hist_final_pos_cm.png"))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=args.dpi)
    labels = ["success", "near-miss", "drop", "stuck"]; counts = [n_succ, n_near, n_drop, n_stuck]
    ax.bar(labels, counts); ax.set_title("Episode outcomes")
    for i, c in enumerate(counts): ax.text(i, c, str(c), ha="center", va="bottom")
    savefig(fig, os.path.join(out_dir, "bar_outcomes.png"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
    ax.scatter(df.loc[succ_mask, "length"], df.loc[succ_mask, "return"], s=10, label="success")
    ax.scatter(df.loc[near_mask, "length"], df.loc[near_mask, "return"], s=10, label="near-miss")
    ax.scatter(df.loc[drop_mask, "length"], df.loc[drop_mask, "return"], s=10, label="drop")
    ax.scatter(df.loc[stuck_mask, "length"], df.loc[stuck_mask, "return"], s=10, label="stuck")
    ax.set_xlabel("episode length (steps)"); ax.set_ylabel("return"); ax.set_title("Return vs. length"); ax.legend(markerscale=2)
    savefig(fig, os.path.join(out_dir, "scatter_return_vs_length.png"))

    if (df["success"] == 1).any():
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
        tts = df.loc[df["success"] == 1, "time_to_success_steps"].replace(-1, np.nan).dropna()
        safe_hist(ax, tts, bins=40, title="Time to first success (steps)", xlabel="time_to_success_steps")
        savefig(fig, os.path.join(out_dir, "hist_tts_success.png"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
    if df["num_goal_hits"].notna().any():
        max_hits = int(df["num_goal_hits"].fillna(0).max())
        bins = range(max_hits + 2)
    else:
        bins = 10
    safe_hist(ax, df["num_goal_hits"], bins=bins, title="Goal hits per episode", xlabel="num_goal_hits")
    savefig(fig, os.path.join(out_dir, "hist_hits_per_episode.png"))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=args.dpi)
    env_hits = df.groupby("env_id")["num_goal_hits"].sum().sort_index()
    env_hits.plot(kind="bar", ax=ax); ax.set_title("Total goal hits per env_id")
    ax.set_xlabel("env_id"); ax.set_ylabel("hits")
    savefig(fig, os.path.join(out_dir, "bar_hits_per_env.png"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
    ax.scatter(df["min_rot_err_deg"], df["return"], s=10)
    add_tol_lines(ax, succ_deg, near_hi); ax.legend()
    ax.set_xlabel("min_rot_err_deg"); ax.set_ylabel("return"); ax.set_title("Return vs min rotation error")
    savefig(fig, os.path.join(out_dir, "scatter_min_rot_vs_return.png"))

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=args.dpi)
    cum_succ = (df["success"] == 1).astype(int).cumsum()
    ax.plot(df["episode"], cum_succ)
    ax.set_xlabel("episode"); ax.set_ylabel("cumulative successes"); ax.set_title("Cumulative successes over episodes")
    savefig(fig, os.path.join(out_dir, "line_cumulative_successes.png"))

    window = min(20, len(df))
    if window >= 2:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=args.dpi)
        roll_s = (df["success"] == 1).astype(float).rolling(window).mean() * 100.0
        roll_d = (df["dropped"] == 1).astype(float).rolling(window).mean() * 100.0
        ax.plot(df["episode"], roll_s, label=f"success rate (rolling {window})")
        ax.plot(df["episode"], roll_d, label=f"drop rate (rolling {window})")
        ax.set_xlabel("episode"); ax.set_ylabel("rate (%)"); ax.set_title("Rolling success & drop rates"); ax.legend()
        savefig(fig, os.path.join(out_dir, "line_rolling_rates.png"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
    data = [
        df.loc[succ_mask, "min_rot_err_deg"].dropna(),
        df.loc[near_mask, "min_rot_err_deg"].dropna(),
        df.loc[drop_mask, "min_rot_err_deg"].dropna(),
        df.loc[stuck_mask, "min_rot_err_deg"].dropna(),
    ]
    ax.boxplot(data, labels=["success","near-miss","drop","stuck"], showfliers=False)
    ax.set_title("Min rotation error by outcome"); ax.set_ylabel("min_rot_err_deg")
    savefig(fig, os.path.join(out_dir, "box_min_rot_by_outcome.png"))

    # Original 2x2 dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=args.dpi)
    ax = axes[0, 0]
    safe_hist(ax, df["min_rot_err_deg"], bins=40, title="Min rotation error (deg)", xlabel="min_rot_err_deg")
    add_tol_lines(ax, succ_deg, near_hi); ax.legend()

    ax = axes[0, 1]
    safe_hist(ax, df["final_rot_err_deg"], bins=40, title="Final rotation error (deg)", xlabel="final_rot_err_deg")
    add_tol_lines(ax, succ_deg, near_hi); ax.legend()

    ax = axes[1, 0]
    labels = ["Success", "Near miss", "Drop", "Stuck"]; counts = [n_succ, n_near, n_drop, n_stuck]
    ax.bar(labels, counts); ax.set_title("Episode outcomes")
    for i, c in enumerate(counts): ax.text(i, c, str(c), ha="center", va="bottom")

    ax = axes[1, 1]
    ax.scatter(df.loc[succ_mask, "length"], df.loc[succ_mask, "return"], s=10, label="success")
    ax.scatter(df.loc[near_mask, "length"], df.loc[near_mask, "return"], s=10, label="near-miss")
    ax.scatter(df.loc[drop_mask, "length"], df.loc[drop_mask, "return"], s=10, label="drop")
    ax.scatter(df.loc[stuck_mask, "length"], df.loc[stuck_mask, "return"], s=10, label="stuck")
    ax.set_xlabel("episode length (steps)"); ax.set_ylabel("return")
    ax.set_title("Return vs. length"); ax.legend(markerscale=2)

    fig.suptitle(f"Eval: {os.path.basename(args.csv_path)}", y=0.995)
    savefig(fig, os.path.join(out_dir, "evaluation_plots.png"))

    print("[INFO] Saved all figures to:", out_dir)

if __name__ == "__main__":
    main()

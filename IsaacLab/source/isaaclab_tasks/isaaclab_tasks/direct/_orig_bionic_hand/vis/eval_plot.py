#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Font size configuration variables
TITLE_FONTSIZE = 16
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 13


def outcome_masks(df, succ_deg, near_hi):
    """Return boolean masks for success, drop, and other (everything else)."""
    succ = (df["success"] == 1)
    drop = (df["dropped"] == 1)
    other = (~succ) & (~drop)
    return succ, drop, other


def add_tol_lines(ax, succ_deg, near_hi, label_prefix=""):
    ax.axvline(succ_deg, linestyle="--", linewidth=2, label=f"{label_prefix}success tol = {succ_deg:.2f}°")
    ax.axvspan(succ_deg, near_hi, alpha=0.2, label=f"{label_prefix}near +{near_hi - succ_deg:.1f}°")


def safe_hist(ax, series, bins, title, xlabel):
    s = series.dropna()
    if len(s) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=TITLE_FONTSIZE)
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("#episodes", fontsize=LABEL_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        return
    ax.hist(s, bins=bins)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("#episodes", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Plot IsaacLab eval CSV with success/drop coloring and tolerance lines.")
    p.add_argument("csv_path", type=str, help="Path to evaluation.csv")
    p.add_argument("--success_tol_rad", type=float, default=None,
                   help="Env success tolerance in radians (e.g., 0.4). If set, overrides --success_tol_deg.")
    p.add_argument("--success_tol_deg", type=float, default=None,
                   help="Env success tolerance in degrees. Ignored if --success_tol_rad is set.")
    p.add_argument("--near_extra_deg", type=float, default=10.0,
                   help="Visual band above success tolerance (degrees).")
    p.add_argument("--dpi", type=int, default=140)
    p.add_argument("--max_episodes", type=int, default=None,
                   help="Only use first N rows.")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Where to save plots/CSVs. Default: <csv_dir>/eval_plots")
    p.add_argument("--file_prefix", type=str, default="", help="Prefix to prepend to all saved plot filenames.")
    args = p.parse_args()

    df = pd.read_csv(args.csv_path)
    if args.max_episodes is not None:
        df = df.iloc[: args.max_episodes].copy()

    needed = [
        "episode","env_id","return","length","final_rot_err_deg","min_rot_err_deg",
        "final_pos_err_cm","success","time_to_success_steps","num_goal_hits","dropped"
    ]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"CSV missing column: {c}")

    df["success"] = df["success"].fillna(0).astype(int)
    df["dropped"] = df["dropped"].fillna(0).astype(int)
    for col in [
        "final_rot_err_deg","min_rot_err_deg","final_pos_err_cm","return",
        "length","time_to_success_steps","num_goal_hits"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # success tolerance in deg (for vertical lines)
    if args.success_tol_rad is not None:
        succ_deg = float(np.degrees(args.success_tol_rad))
    elif args.success_tol_deg is not None:
        succ_deg = float(args.success_tol_deg)
    else:
        succ_deg = float(np.degrees(0.4))  # default matches env (0.4 rad)
    near_hi = succ_deg + float(args.near_extra_deg)

    # output folder
    csv_dir = os.path.dirname(os.path.abspath(args.csv_path))
    out_dir = args.out_dir or os.path.join(csv_dir, "eval_plots")
    os.makedirs(out_dir, exist_ok=True)

    # prefix for file names
    prefix = args.file_prefix

    # masks (success, drop, other)
    succ_mask, drop_mask, other_mask = outcome_masks(df, succ_deg, near_hi)

    # console summary
    n = len(df)
    n_succ = int(succ_mask.sum())
    n_drop = int(drop_mask.sum())
    n_other = int(other_mask.sum())
    print("\n[INFO] Evaluation summary")
    print(f"[INFO] Episodes:                   {n}")
    print(f"[INFO] Success tolerance (visual): {succ_deg:.3f}° ({np.radians(succ_deg):.3f} rad)")
    if n > 0:
        print(f"[INFO] Successes:                  {n_succ} ({100.0 * n_succ / n:.1f}%)")
        print(f"[INFO] Drops:                      {n_drop} ({100.0 * n_drop / n:.1f}%)")
        print(f"[INFO] Other (not success/drop):   {n_other} ({100.0 * n_other / n:.1f}%)")
    else:
        print("[INFO] No episodes to report.")
    print(f"[INFO] Saving outputs to: {out_dir}")

    # CSVs in folder
    outcome = np.where(succ_mask, "success", np.where(drop_mask, "drop", "other"))
    out_ep = df.copy()
    out_ep["outcome"] = outcome
    out_ep.to_csv(os.path.join(out_dir, f"{prefix}episode_outcomes.csv"), index=False)

    # ---- Separate plots ----
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
    safe_hist(ax, df["min_rot_err_deg"], bins=40, title="Min rotation error per episode (deg)", xlabel="min_rot_err_deg")
    add_tol_lines(ax, succ_deg, near_hi); ax.legend(fontsize=LEGEND_FONTSIZE)
    savefig(fig, os.path.join(out_dir, f"{prefix}hist_min_rot_deg.png"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
    safe_hist(ax, df["final_rot_err_deg"], bins=40, title="Final rotation error per episode (deg)", xlabel="final_rot_err_deg")
    add_tol_lines(ax, succ_deg, near_hi); ax.legend(fontsize=LEGEND_FONTSIZE)
    savefig(fig, os.path.join(out_dir, f"{prefix}hist_final_rot_deg.png"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
    safe_hist(ax, df["final_pos_err_cm"], bins=40, title="Final position error per episode (cm)", xlabel="final_pos_err_cm")
    savefig(fig, os.path.join(out_dir, f"{prefix}hist_final_pos_cm.png"))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=args.dpi)
    labels = ["success", "drop", "other"]; counts = [n_succ, n_drop, n_other]
    ax.bar(labels, counts)
    ax.set_title("Episode outcomes", fontsize=TITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=LABEL_FONTSIZE)
    savefig(fig, os.path.join(out_dir, f"{prefix}bar_outcomes.png"))

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=args.dpi)
    ax.scatter(df.loc[succ_mask, "length"], df.loc[succ_mask, "return"], s=18, label=f"success (n={n_succ})")
    ax.scatter(df.loc[drop_mask, "length"], df.loc[drop_mask, "return"], s=18, label=f"drop (n={n_drop})")
    ax.scatter(df.loc[other_mask, "length"], df.loc[other_mask, "return"], s=18, label=f"other (n={n_other})")
    ax.set_xlabel("episode length (steps)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("return", fontsize=LABEL_FONTSIZE)
    ax.set_title("Return vs. length", fontsize=TITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    ax.legend(markerscale=2, fontsize=LEGEND_FONTSIZE)
    savefig(fig, os.path.join(out_dir, f"{prefix}scatter_return_vs_length.png"))

    if (df["success"] == 1).any():
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
        tts = df.loc[df["success"] == 1, "time_to_success_steps"].replace(-1, np.nan).dropna()
        safe_hist(ax, tts, bins=40, title="Time to first success (steps)", xlabel="time_to_success_steps")
        savefig(fig, os.path.join(out_dir, f"{prefix}hist_tts_success.png"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
    if df["num_goal_hits"].notna().any():
        max_hits = int(df["num_goal_hits"].fillna(0).max())
        bins = range(max_hits + 2)
    else:
        bins = 10
    safe_hist(ax, df["num_goal_hits"], bins=bins, title="Goal hits per episode", xlabel="num_goal_hits")
    savefig(fig, os.path.join(out_dir, f"{prefix}hist_hits_per_episode.png"))

    # Return vs min rotation error colored by outcome (success / drop / other)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=args.dpi)
    ax.scatter(df.loc[succ_mask, "min_rot_err_deg"], df.loc[succ_mask, "return"], s=10, label="success")
    ax.scatter(df.loc[drop_mask, "min_rot_err_deg"], df.loc[drop_mask, "return"], s=10, label="drop")
    ax.scatter(df.loc[other_mask, "min_rot_err_deg"], df.loc[other_mask, "return"], s=10, label="other")
    add_tol_lines(ax, succ_deg, near_hi)
    ax.legend(markerscale=2, fontsize=LEGEND_FONTSIZE)
    ax.set_xlabel("min_rot_err_deg", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("return", fontsize=LABEL_FONTSIZE)
    ax.set_title("Return vs min rotation error", fontsize=TITLE_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)
    savefig(fig, os.path.join(out_dir, f"{prefix}scatter_min_rot_vs_return.png"))

    # REMOVED: cumulative successes plot & total hits per env_id plot & box plot & evaluation_plots

    window = min(20, len(df))
    if window >= 2:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4), dpi=args.dpi)
        roll_s = (df["success"] == 1).astype(float).rolling(window).mean() * 100.0
        roll_d = (df["dropped"] == 1).astype(float).rolling(window).mean() * 100.0
        ax.plot(df["episode"], roll_s, label=f"success rate (rolling {window})")
        ax.plot(df["episode"], roll_d, label=f"drop rate (rolling {window})")
        ax.set_xlabel("episode", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("rate (%)", fontsize=LABEL_FONTSIZE)
        ax.set_title("Rolling success & drop rates", fontsize=TITLE_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)
        ax.legend(fontsize=LEGEND_FONTSIZE)
        savefig(fig, os.path.join(out_dir, f"{prefix}line_rolling_rates.png"))

    print("[INFO] Saved all figures to:", out_dir)


if __name__ == "__main__":
    main()

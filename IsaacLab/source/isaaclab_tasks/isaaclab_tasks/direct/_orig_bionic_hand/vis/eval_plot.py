import os, pandas as pd, numpy as np, matplotlib.pyplot as plt

RUNS = {
    "logs/.../ShadowHand/evaluation.csv": "Shadow Hand",
    "logs/.../BionicHand-std/evaluation.csv": "Bionic Hand (std)",
    "logs/.../BionicHand+Wrist/evaluation.csv": "Bionic + Wrist",
    "logs/.../BionicHand-ROMx/evaluation.csv": "Bionic (ROM↑)",
}

dfs = []
for path, label in RUNS.items():
    df = pd.read_csv(path)
    df["config"] = label
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

def med_iqr(x):
    return f"{np.nanmedian(x):.2f} [{np.nanpercentile(x,25):.2f}, {np.nanpercentile(x,75):.2f}]"

# ----- summary table -----
summary = data.groupby("config").apply(lambda g: pd.Series({
    "Success (%)": 100.0 * g["success"].mean(),
    "Final err (°) med[IQR]": med_iqr(g["final_rot_err_deg"]),
    "Min err (°) med[IQR]": med_iqr(g["min_rot_err_deg"]),
    "Drop rate (%)": 100.0 * g["dropped"].mean(),
    "TTS (steps) med[IQR] (succ only)": med_iqr(g.loc[g["success"]==1, "time_to_success_steps"]),
    "n_episodes": len(g),
}))
print(summary.to_string())

# ----- Success rate bar -----
plt.figure()
suc = data.groupby("config")["success"].mean().sort_values()
(suc * 100).plot(kind="bar")
plt.ylabel("Success (%)")
plt.title("Success rate (env success_tolerance=0.4)")
plt.tight_layout()
plt.show()

# ----- ECDF of min rotation error -----
plt.figure()
for cfg, g in data.groupby("config"):
    x = np.sort(g["min_rot_err_deg"].values)
    y = np.linspace(0, 1, len(x), endpoint=True)
    plt.plot(x, y, label=cfg)
plt.xlabel("Minimum orientation error within episode (deg)")
plt.ylabel("ECDF")
plt.legend()
plt.tight_layout()
plt.show()

# ----- Final rotation error distribution -----
plt.figure()
order = data["config"].unique()
data.boxplot(column="final_rot_err_deg", by="config", grid=False)
plt.suptitle("")
plt.xlabel("")
plt.ylabel("Final orientation error (deg)")
plt.tight_layout()
plt.show()

# ----- Optional: simple KM-style survival (success as event) -----
# Requires per-episode lengths and censoring; skip here if not logged cleanly.

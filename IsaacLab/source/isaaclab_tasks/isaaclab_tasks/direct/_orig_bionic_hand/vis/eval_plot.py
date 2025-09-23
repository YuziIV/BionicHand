# This script visualizes key metrics from an RL evaluation CSV file
# and saves the generated plots as an image file.

import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Set the full path to your CSV file
csv_file_path = "/workspace/isaaclab/logs/skrl/bionic_hand/2025-09-23_01-50-23_ppo_torch/evaluation.csv"

# Get the directory of the CSV file to save the plot image in the same location
output_dir = os.path.dirname(csv_file_path)
output_path = os.path.join(output_dir, 'evaluation_plots.png')

# --- Data Loading and Pre-processing ---
try:
    # Read the data from the CSV file into a pandas DataFrame.
    df = pd.read_csv(csv_file_path)
    # Filter for successful episodes to plot errors.
    successful_episodes = df[df['success'] == 1]
except FileNotFoundError:
    print(f"Error: The file '{csv_file_path}' was not found.")
    exit()
except pd.errors.EmptyDataError:
    print(f"Error: The file '{csv_file_path}' is empty.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

# --- Plotting ---
# Create a figure and a set of subplots.
# The layout is 2 rows by 2 columns.
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('RL Agent Performance Evaluation', fontsize=20, y=1.02)
plt.style.use('seaborn-v0_8-whitegrid')

# Plot 1: Episode Return vs. Episode Number
ax1 = axes[0, 0]
ax1.scatter(df['episode'], df['return'], alpha=0.6, s=15, color='royalblue')
ax1.set_title('Episode Return vs. Episode Number', fontsize=14)
ax1.set_xlabel('Episode')
ax1.set_ylabel('Total Return')

# Plot 2: Success Rate (Moving Average)
ax2 = axes[0, 1]
# Calculate a rolling average of the success rate over a window of 25 episodes
rolling_success_rate = df['success'].rolling(window=25, min_periods=1).mean()
ax2.plot(df['episode'], rolling_success_rate, color='seagreen', linewidth=2)
ax2.set_title('Success Rate (25-Episode Rolling Average)', fontsize=14)
ax2.set_xlabel('Episode')
ax2.set_ylabel('Success Rate')
ax2.set_ylim(-0.05, 1.05)

# Plot 3: Distribution of Final Rotation Error for Successful Episodes
ax3 = axes[1, 0]
if not successful_episodes.empty:
    ax3.hist(successful_episodes['final_rot_err_deg'], bins=20, color='indianred', edgecolor='black')
    ax3.set_title('Final Rotation Error for Successful Episodes', fontsize=14)
    ax3.set_xlabel('Final Rotation Error (degrees)')
    ax3.set_ylabel('Frequency')
else:
    ax3.text(0.5, 0.5, 'No successful episodes to plot.', ha='center', va='center', fontsize=12)
    ax3.set_title('Final Rotation Error for Successful Episodes', fontsize=14)

# Plot 4: Distribution of Final Position Error for Successful Episodes
ax4 = axes[1, 1]
if not successful_episodes.empty:
    ax4.hist(successful_episodes['final_pos_err_cm'], bins=20, color='darkorange', edgecolor='black')
    ax4.set_title('Final Position Error for Successful Episodes', fontsize=14)
    ax4.set_xlabel('Final Position Error (cm)')
    ax4.set_ylabel('Frequency')
else:
    ax4.text(0.5, 0.5, 'No successful episodes to plot.', ha='center', va='center', fontsize=12)
    ax4.set_title('Final Position Error for Successful Episodes', fontsize=14)

# Adjust layout to prevent titles and labels from overlapping.
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the figure to a file.
plt.savefig(output_path, dpi=300)

print(f"Plots generated successfully and saved to '{output_path}'.")

# Close the plot figure to free up resources.
plt.close(fig)

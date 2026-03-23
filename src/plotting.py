import matplotlib.pyplot as plt
import os
import numpy as np

def plot_results(df, target_abundance, folder, filename):
    os.makedirs(folder, exist_ok=True)  # ensure results folder exists
    fig, ax1 = plt.subplots(figsize=(10, 6))  # create main plot

    # Avoid log(0) errors
    true_abundance = df["True_Abundance"].clip(lower=1e-1)  # clip minimum value
    observed_abundance = df["Observed_Abundance"].clip(lower=1e-1)

    # Plot true and observed abundances
    ax1.plot(df["Time"], true_abundance, color='blue', label="True Abundance")
    ax1.plot(df["Time"], observed_abundance, color='cyan', linestyle='--', label="Observed Abundance (Survey)")
    ax1.axhline(target_abundance, linestyle="--", color='green', alpha=0.5, label="Target Abundance")  # target line

    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Abundance", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.set_yscale('log')  # use log scale for abundance

    # Set safe y-limits for abundance
    min_ab = true_abundance.min()
    max_ab = true_abundance.max()
    if min_ab == max_ab:  # prevent zero range
        max_ab = min_ab * 10
    ax1.set_ylim(bottom=max(min_ab * 0.8, 1e-1), top=max_ab * 1.2)

    # Secondary axis for fishing effort
    ax2 = ax1.twinx()
    effort = df["Effort"]
    ax2.plot(df["Time"], effort, color='red', label="Fishing Effort")
    ax2.set_ylabel("Fishing Effort", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Set safe y-limits for effort
    min_eff = effort.min()
    max_eff = effort.max()
    if max_eff <= 0 or min_eff == max_eff:  # prevent zero range
        max_eff = max(1.0, max_eff + 1.0)
    ax2.set_ylim(bottom=0, top=max_eff * 1.2)

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    fig.tight_layout()  # adjust layout
    save_path = os.path.join(folder, filename)  # final path
    fig.savefig(save_path, dpi=150)  # save figure
    plt.close(fig)  # close plot to free memory
    return save_path  # return file path

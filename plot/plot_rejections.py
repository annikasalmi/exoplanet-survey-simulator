import os
import numpy as np
import matplotlib.pyplot as plt

from plot.helpers import make_output_dir, get_rejection_reason
from tools.constants import iwa, min_planet_flux_star_ratio, min_photons


def plot_failures_piechart(df, nruns=1, star_catalog='Gaia',name='hwo'):
 
    df['rejection_reason'] = df.apply(get_rejection_reason, axis=1)

    # Step 2: Count reasons
    counts = df['rejection_reason'].value_counts()

    # Step 3: Plot pie chart
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c']  # red, orange, blue, green
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=colors[:len(counts)])
    plt.title(f"{name}\nTotal planets: {len(df)}")
    plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
    plt.tight_layout()

    data_dir = make_output_dir(name, nruns, star_catalog)

    plt.savefig(os.path.join(data_dir, f"failure_detected_{name}_nruns{nruns}_{star_catalog}.png"), 
                             dpi=300, bbox_inches='tight')
    
def plot_failures_histogram_multipanel(df, nruns=1, star_catalog='Gaia', name='hwo'):
    df['rejection_reason'] = df.apply(get_rejection_reason, axis=1)

    # Define failure categories and thresholds
    cutoffs = {
        'Min Photons': ('photon_rate', min_photons),
        'Flux Ratio': ('flux_ratio', min_planet_flux_star_ratio),
        'IWA': ('angsep', iwa),
    }

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, (reason, (column, threshold)) in zip(axs, cutoffs.items()):
        reason_df = df[df['rejection_reason'] == reason]

        ax.hist(df[column], bins=40, color='lightgray', edgecolor='black', log=True)

        if isinstance(threshold, tuple):
            ax.axvline(threshold[0], color='red', linestyle='--', label=f'Min cutoff = {threshold[0]:.2e}')
            ax.axvline(threshold[1], color='red', linestyle='--', label=f'Max cutoff = {threshold[1]:.2e}')
        else:
            ax.axvline(threshold, color='red', linestyle='--', label=f'Cutoff = {threshold:.2e}')

        ax.set_title(f"{reason} — {len(reason_df)} failures")
        ax.set_xlabel(column.replace('_', ' ').capitalize())
        ax.set_ylabel("Number of planets")
        ax.legend()

    plt.suptitle("Reasons for Planet Non-Detection", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    data_dir = make_output_dir(name, nruns, star_catalog)
    filename = f"failure_multipanel_{name}_nruns{nruns}_{star_catalog}.png"
    plt.savefig(os.path.join(data_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    
def plot_rejections(df, nruns=1, star_catalog='Gaia', name='hwo'):
    plot_failures_histogram(df, nruns=nruns, star_catalog=star_catalog, name=name)
    plot_failures_piechart(df, nruns=nruns, star_catalog=star_catalog, name=name)
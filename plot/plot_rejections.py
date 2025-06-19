import os
import numpy as np
import matplotlib.pyplot as plt

from plot.helpers import make_output_dir, get_rejection_reason
from tools.constants import iwa, min_planet_flux_star_ratio, min_flux


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
    

def plot_failures_histogram(df, nruns=1, star_catalog='Gaia', name='hwo'):
    df['rejection_reason'] = df.apply(get_rejection_reason, axis=1)

    # === Histograms for Each Failure Reason ===
    cutoffs = {
        'Min Photons': ('photon_rate', min_flux),
        'Flux Ratio': ('flux_ratio', min_planet_flux_star_ratio),
        'IWA': ('angsep', iwa),
    }

    for reason, (column, threshold) in cutoffs.items():
        reason_df = df[df['rejection_reason'] == reason]

        plt.figure(figsize=(8, 5))
        plt.hist(df[column], bins=40, color='lightgray', edgecolor='black', log=True)

        if isinstance(threshold, tuple):
            # Draw two lines for a range
            plt.axvline(threshold[0], color='red', linestyle='--', label='Min cutoff = {:.2e}'.format(threshold[0]))
            plt.axvline(threshold[1], color='red', linestyle='--', label='Max cutoff = {:.2e}'.format(threshold[1]))
        else:
            # Single cutoff
            plt.axvline(threshold, color='red', linestyle='--', label='Cutoff = {:.2e}'.format(threshold))

        plt.title(f"Reason planet not detected: {reason} too low — {len(reason_df)} cases")
        plt.xlabel(column.replace('_', ' ').capitalize())
        plt.ylabel("Number of planets")
        plt.legend()
        plt.tight_layout()

        data_dir = make_output_dir(name, nruns, star_catalog)

        filename = f"failure_{reason.replace(' ', '_')}_{name}_nruns{nruns}_{star_catalog}.png"
        plt.savefig(os.path.join(data_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    
def plot_rejections(df, nruns=1, star_catalog='Gaia', name='hwo'):
    plot_failures_histogram(df, nruns=nruns, star_catalog=star_catalog, name=name)
    plot_failures_piechart(df, nruns=nruns, star_catalog=star_catalog, name=name)
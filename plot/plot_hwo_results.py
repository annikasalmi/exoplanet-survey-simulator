import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools.paths import LIFESIM_DATA_DIR

def plot(results, nruns=1, star_catalog='Gaia'):
    # Combine all run results
    df_all = pd.concat(results, keys=range(nruns)).reset_index(level=0).rename(columns={'level_0': 'run'})

    # Pivot to get one row per run, one column per (stype, radius_bin)
    df_pivot = df_all.pivot_table(index='run', columns=['stype', 'radius_bin'], values='count', fill_value=0)

    # Compute mean and std across runs
    df_mean = df_pivot.mean(axis=0)
    df_std = df_pivot.std(axis=0)

    # Convert MultiIndex back to DataFrame
    grouped_sum = df_mean.reset_index(name='count')
    grouped_sum['error'] = df_std.values

    # Pivot to plotting format
    pivot_counts = grouped_sum.pivot(index='stype', columns='radius_bin', values='count').fillna(0)
    pivot_errors = grouped_sum.pivot(index='stype', columns='radius_bin', values='error').fillna(0)

    # Reorder for clean plotting
    star_order = ['F', 'G', 'K', 'M']
    bin_labels = ['<1.5', '1.5–3.0', '3.0–6.0', 'Rocky HZ']
    pivot_counts = pivot_counts.reindex(star_order).reindex(columns=bin_labels, fill_value=0)
    pivot_errors = pivot_errors.reindex(star_order).reindex(columns=bin_labels, fill_value=0)

    # Create grouped bar plot
    colors = ['lightblue', 'deepskyblue', 'midnightblue', 'forestgreen']
    hatches = ['...', 'ooo', 'OO', None]
    bar_width = 0.2
    x = np.arange(len(star_order))

    _, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(bin_labels):
        heights = pivot_counts[label].tolist()
        errors = pivot_errors[label].tolist()

        ax.bar(x + i * bar_width, heights, width=bar_width,
               yerr=errors, label=label,
               color=colors[i], hatch=hatches[i], edgecolor='black')

        # Add text annotations
        for j, (h, err) in enumerate(zip(heights, errors)):
            ax.text(x[j] + i * bar_width, h + 2, f"{int(h)}±{int(err)}", ha='center', fontsize=8)

    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(star_order)
    ax.set_ylabel('Detectable Planets')
    ax.set_title('Detectable Planets by Star Type')
    ax.legend(title='Planet Radius')
    plt.tight_layout()

    plt.savefig(os.path.join(HWO_DATA_DIR,f"planets_hwo_nruns{nruns}_{star_catalog}.png"), 
                             dpi=300, bbox_inches='tight')
    plt.show()
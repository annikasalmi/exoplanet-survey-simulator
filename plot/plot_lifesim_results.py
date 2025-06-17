import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools.paths import LIFESIM_DATA_DIR

def plot(results, nruns, star_catalog='Gaia'):
    # Parameters
    nruns = len(results)  # List of DataFrames, one per run
    star_order = ['F', 'G', 'K', 'M']
    bin_labels = ['<1.5', '1.5–3.0', '3.0–6.0', 'Rocky HZ']

    # Step 1: Process each run
    processed_runs = []

    for run_idx, df in enumerate(results):
        df_detected = df[df['detected'] == True].copy()
        
        bins = [0, 1.5, 3.0, 6.0]
        labels = ['<1.5', '1.5–3.0', '3.0–6.0']
        df_detected['radius_bin'] = pd.cut(df_detected['radius_p'], bins=bins, labels=labels, include_lowest=True)

        # Add "Rocky HZ" bin
        rocky_hz = df[(df['habitable'] == True) & (df['radius_p'] < 1.5)].copy()
        rocky_hz['radius_bin'] = 'Rocky HZ'

        # Group by stype and radius bin
        grouped = df_detected.groupby(['stype', 'radius_bin']).size().reset_index(name='count')
        grouped['run'] = run_idx
        processed_runs.append(grouped)

    # Step 2: Combine all runs into one DataFrame
    df_all = pd.concat(processed_runs, ignore_index=True)

    # Step 3: Pivot: one row per run, columns = (stype, radius_bin)
    df_pivot = df_all.pivot_table(index='run', columns=['stype', 'radius_bin'], values='count', fill_value=0)

    # Step 4: Mean and std across runs
    df_mean = df_pivot.mean(axis=0)
    df_std = df_pivot.std(axis=0)

    # Step 5: Flatten MultiIndex back into DataFrame
    grouped_sum = df_mean.reset_index(name='count')
    grouped_sum['error'] = df_std.values

    # Step 6: Pivot into plotting format
    pivot_counts = grouped_sum.pivot(index='stype', columns='radius_bin', values='count').fillna(0)
    pivot_errors = grouped_sum.pivot(index='stype', columns='radius_bin', values='error').fillna(0)

    # Step 7: Reorder for clean plotting
    pivot_counts = pivot_counts.reindex(star_order).reindex(columns=bin_labels, fill_value=0)
    pivot_errors = pivot_errors.reindex(star_order).reindex(columns=bin_labels, fill_value=0)

    # Step 8: Plotting
    colors = ['lightblue', 'deepskyblue', 'midnightblue', 'forestgreen']
    hatches = ['...', 'ooo', 'OO', None]
    bar_width = 0.2
    x = np.arange(len(star_order))

    fig, ax = plt.subplots(figsize=(10, 6))

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

    # Save and show
    plt.savefig(os.path.join(LIFESIM_DATA_DIR,f"planets_lifesim_nruns{nruns}_{star_catalog}.png"), 
                             dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Plot saved to {os.path.join(LIFESIM_DATA_DIR,f'planets_lifesim_nruns{nruns}_{star_catalog}.png')}")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools.paths import HWO_DATA_DIR

def plot_by_star(df_all, nruns=1, star_catalog='Gaia'):
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

    plt.savefig(os.path.join(HWO_DATA_DIR,f"planets_stellar_type_hwo_nruns{nruns}_{star_catalog}.png"), 
                             dpi=300, bbox_inches='tight')
    plt.show()

def temp_zone(temp):
    if temp > 600:
        return 'hot'
    elif temp > 300:
        return 'warm'
    else:
        return 'cold'

def assign_category(row):
    r = row['radius']
    hab = row['habitable']
    stype = row['star_type']

    if r < 1.5 and hab:
        return 'Rocky eHZ'
    elif r < 1.8 and hab and stype in ['G', 'K']:
        return 'Exo-Earth Candidates'
    elif 1.0 <= r < 2.0:
        return 'Rocky + Super-Earths'
    elif 2.0 <= r < 4.0:
        return 'Sub-Neptunes'
    elif 4.0 <= r < 8.0:
        return 'Sub-Jovians'
    else:
        return None

def plot_by_planet(df, nruns=1, star_catalog='Gaia'):
    # Apply binning
    df['temp_zone'] = df['temperature'].apply(temp_zone)
    df['category'] = df.apply(assign_category, axis=1)
    df = df.dropna(subset=['category'])

    # Count by category and temp zone
    grouped = df.groupby(['category', 'temp_zone']).size().unstack(fill_value=0)

    # Ensure consistent order
    categories = ['Rocky eHZ', 'Exo-Earth Candidates', 'Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
    temp_zones = ['hot', 'warm', 'cold']
    grouped = grouped.reindex(index=categories, columns=temp_zones, fill_value=0)

    # Plot
    x = np.arange(len(categories))
    bar_width = 0.6

    hot_vals = grouped['hot'].values
    warm_vals = grouped['warm'].values
    cold_vals = grouped['cold'].values

    fig, ax = plt.subplots(figsize=(9, 6))

    p1 = ax.bar(x, hot_vals, bar_width, label='hot', color='red', hatch='//', edgecolor='black')
    p2 = ax.bar(x, warm_vals, bar_width, bottom=hot_vals, label='warm', color='gold', hatch='--', edgecolor='black')
    p3 = ax.bar(x, cold_vals, bar_width, bottom=hot_vals + warm_vals, label='cold', color='blue', hatch='\\\\', edgecolor='black')

    totals = hot_vals + warm_vals + cold_vals
    for i, val in enumerate(totals):
        ax.text(x[i], val + 2, str(val), ha='center', va='bottom', fontsize=10)

    ax.set_ylabel("Detectable planets")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper left')

    # Optional: annotation box
    textstr = 'D = 2.0 m\nScenario 1'
    props = dict(boxstyle='round', facecolor='white', edgecolor='black')
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    plt.savefig(os.path.join(HWO_DATA_DIR,f"planets_planet_type_hwo_nruns{nruns}_{star_catalog}.png"), 
                             dpi=300, bbox_inches='tight')
    plt.show()
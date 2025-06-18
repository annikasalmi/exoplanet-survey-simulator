import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools.paths import HWO_PLOTS_DIR, LIFESIM_DATA_DIR

def plot_by_star(df, nruns=1, star_catalog='Gaia', name='hwo'):
    '''
    Plots the number of detectable planets by star type and radius bin.
    '''
    # Group by run, stype, radius_bin
    df = df.groupby(['run', 'stype', 'radius_bin']).size().reset_index(name='count')

    # Pivot to have runs as index and (stype, radius_bin) as columns
    df_pivot = df.pivot_table(index='run', columns=['stype', 'radius_bin'], values='count', fill_value=0)

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
    ax.set_title(f'Detectable Planets by Star Type for {name} for {nruns} Runs\nStar Catalog: {star_catalog}')
    ax.legend(title='Planet Radius')

    plt.tight_layout()
    if name == 'hwo':
        name = 'HWO'
        data_dir = HWO_PLOTS_DIR
    if name == 'lifesim':
        name = 'LIFEsim'
        data_dir = LIFESIM_DATA_DIR

    plt.savefig(os.path.join(data_dir, f"stellar_type_{name}_nruns{nruns}_{star_catalog}.png"), 
                             dpi=300, bbox_inches='tight')

def temp_zone(temp):
    '''
    Assigns a temperature zone based on the temperature value.'''
    if temp > 600:
        return 'hot'
    elif temp > 300:
        return 'warm'
    else:
        return 'cold'

def assign_category(row):
    '''
    Assigns a category based on the planet's radius, habitability, and star type.'''
    r = row['radius_p']
    hab = row['habitable']
    stype = row['stype']

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
    
def plot_by_planet(df, nruns=1, star_catalog='Gaia', name='hwo'):
    '''
    Plots the number of detectable planets by planet type and temperature zone.
    '''
    df['temp_zone'] = df['temp_p'].apply(temp_zone)
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
    bar_width = 0.2  # Smaller width for grouping
    offsets = [-bar_width, 0, bar_width]  # for 'hot', 'warm', 'cold'

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['red', 'gold', 'blue']
    hatches = ['//', '--', '\\\\']
    labels = ['hot', 'warm', 'cold']

    for i, (zone, offset) in enumerate(zip(temp_zones, offsets)):
        values = grouped[zone].values
        ax.bar(x + offset, values, bar_width, label=labels[i],
               color=colors[i], hatch=hatches[i], edgecolor='black')

        # Add text annotations
        for j, val in enumerate(values):
            ax.text(x[j] + offset, val + 1, str(val), ha='center', va='bottom', fontsize=8)

    ax.set_ylabel("Detectable Planets")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=15, ha='right')
    ax.legend(title='Temp Zone')
    ax.set_title(f'Detectable Planets by Planet Type for {name} for {nruns} Runs\nStar Catalog: {star_catalog}')

    # Optional: annotation box
    textstr = 'D = 2.0 m\nScenario 1'
    props = dict(boxstyle='round', facecolor='white', edgecolor='black')
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    if name == 'hwo':
        name = 'HWO'
        data_dir = HWO_PLOTS_DIR
    if name == 'lifesim':
        name = 'LIFEsim'
        data_dir = LIFESIM_DATA_DIR

    plt.savefig(os.path.join(data_dir, f"planet_type_{name}_nruns{nruns}_{star_catalog}.png"), 
                             dpi=300, bbox_inches='tight')
    
def plot_distances(df, nruns=1, star_catalog='Gaia', name='hwo'):
    '''
    Plots the number of detectable planets by distance bins.
    '''
    bins = [0, 3, 5, 7, 9, 11, 13, 15, np.inf]
    labels = ['< 3', '3 - 5', '5 - 7', '7 - 9', '9 - 11', '11 - 13', '13 - 15', '> 15']
    df['distance_bin'] = pd.cut(df['distance_s'], bins=bins, labels=labels, right=False)

    # Count and uncertainty (Poisson: sqrt(N))
    counts = df['distance_bin'].value_counts().reindex(labels, fill_value=0)
    errors = np.sqrt(counts)

    # Plotting
    x = np.arange(len(labels))
    bar_width = 0.6
    color = '#66c2a5'  # Soft green like in your image

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(x, counts, width=bar_width, yerr=errors,
                capsize=4, color=color, hatch='//', edgecolor='black')

    # Add count ± error text
    for i, (val, err) in enumerate(zip(counts, errors)):
        ax.text(i, val + 3, f"{int(val)}±{int(err)}", ha='center', va='bottom', fontsize=10)

    # Labels and formatting
    ax.set_ylabel("Detectable planets", fontsize=12)
    ax.set_xlabel("Distance [pc]", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(counts + errors) * 1.2)
    ax.set_title(f'Detectable Planets by Distance for {name} for {nruns} Runs\nStar Catalog: {star_catalog}')

    plt.tight_layout()
    if name == 'hwo':
        name = 'HWO'
        data_dir = HWO_PLOTS_DIR
    if name == 'lifesim':
        name = 'LIFEsim'
        data_dir = LIFESIM_DATA_DIR

    plt.savefig(os.path.join(data_dir, f"planet_distance_{name}_nruns{nruns}_{star_catalog}.png"), 
                             dpi=300, bbox_inches='tight')
    
def plot_efficiency(df, nruns=1, star_catalog='Gaia', name='hwo'):

    # Filter only rocky eHZ planets
    rocky_ehz = df[df['radius_bin'] == 'Rocky HZ']

    # Bin edges
    bins = np.linspace(125, 305, 40)  # ~4.6 K per bin
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Histogram of all and detected
    total_counts, _ = np.histogram(rocky_ehz['temp_p'], bins=bins)
    detected_counts, _ = np.histogram(rocky_ehz[rocky_ehz['detectable'] == True]['temp_p'], bins=bins)

    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = np.true_divide(detected_counts, total_counts)
        efficiency[np.isnan(efficiency)] = 0.0

    # === Plot ===
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plots
    ax1.bar(bin_centers, total_counts, width=np.diff(bins), align='center', color='lightgrey', label='Rocky, eHZ planets present')
    ax1.bar(bin_centers, detected_counts, width=np.diff(bins), align='center', color='green', label='Rocky, eHZ planets detectable')

    ax1.set_ylabel("Number of rocky, eHZ planets")
    ax1.set_xlabel("Temperature [K]")
    ax1.set_xlim(bins[0], bins[-1])

    # Secondary axis for detection efficiency
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, efficiency, 'r--', linewidth=2, label='Detection efficiency')
    ax2.set_ylabel("Detection efficiency")
    ax2.set_ylim(0, 1.0)

    # Legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left')

    plt.tight_layout()
    if name == 'hwo':
        name = 'HWO'
        data_dir = HWO_PLOTS_DIR
    if name == 'lifesim':
        name = 'LIFEsim'
        data_dir = LIFESIM_DATA_DIR

    plt.savefig(os.path.join(data_dir, f"detection_efficiency_{name}_nruns{nruns}_{star_catalog}.png"), 
                             dpi=300, bbox_inches='tight')
    
def plot_
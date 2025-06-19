import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from plot.helpers import make_output_dir, temp_zone, assign_category, pivot_stats_temp, pivot_stats_radius
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_by_star(df, nruns=1, star_catalog='Gaia', name='hwo'):
       '''
       Creates two grouped bar plots:
       1. Total planet counts by star type and radius bin (stacked detected + undetected with errors).
       2. Detected planets only by star type and radius bin (with error bars).
       '''
       df['detected_flag'] = df['detected'].astype(bool)

       # Group data
       total = df.groupby(['run', 'stype', 'radius_bin']).size().reset_index(name='count')
       detected = df[df['detected_flag']].groupby(['run', 'stype', 'radius_bin']).size().reset_index(name='count')

       total_stats = pivot_stats_radius(total)
       detected_stats = pivot_stats_radius(detected)

       def prep_plot_df(stats, star_order, bin_labels):
              df = stats.pivot(index='stype', columns='radius_bin', values='count').fillna(0)
              errors = stats.pivot(index='stype', columns='radius_bin', values='error').fillna(0)
              df = df.reindex(star_order).reindex(columns=bin_labels, fill_value=0)
              errors = errors.reindex(star_order).reindex(columns=bin_labels, fill_value=0)
              return df, errors

       star_order = ['F', 'G', 'K', 'M']
       bin_labels = ['<1.5', '1.5–3.0', '3.0–6.0', 'Rocky HZ']

       total_df, total_err = prep_plot_df(total_stats, star_order, bin_labels)
       det_df, det_err = prep_plot_df(detected_stats, star_order, bin_labels)

       colors = ['lightblue', 'deepskyblue', 'midnightblue', 'forestgreen']
       hatches = ['...', 'ooo', 'OO', None]
       bar_width = 0.2
       x = np.arange(len(star_order))

       # === Full bar plot with total + detected overlay ===
       fig1, ax1 = plt.subplots(figsize=(10, 6))
       for i, label in enumerate(bin_labels):
              heights = total_df[label].tolist()
              errors = total_err[label].tolist()
              ax1.bar(x + i * bar_width, heights, width=bar_width,
                     yerr=errors, label=label,
                     color=colors[i], hatch=hatches[i], edgecolor='black')
              for j, (h, err) in enumerate(zip(heights, errors)):
                     ax1.text(x[j] + i * bar_width, h + 2, f"{int(h)}±{int(err)}", ha='center', fontsize=8)

       ax1.set_xticks(x + 1.5 * bar_width)
       ax1.set_xticklabels(star_order)
       ax1.set_ylabel('Total Planets')
       ax1.set_title(f'Total Planets by Star Type for {name} ({nruns} Runs)\nStar Catalog: {star_catalog}')
       ax1.legend(title='Radius Bin')
       plt.tight_layout()

       # === Plot detected planets only ===
       _, ax2 = plt.subplots(figsize=(10, 6))
       for i, label in enumerate(bin_labels):
              heights = det_df[label].fillna(0).tolist()
              errors = det_err[label].fillna(0).tolist()
              ax2.bar(x + i * bar_width, heights, width=bar_width,
                     yerr=errors, label=label,
                     color=colors[i], hatch=hatches[i], edgecolor='black')
              for j, (h, err) in enumerate(zip(heights, errors)):
                     ax2.text(x[j] + i * bar_width, h + 2, f"{int(h)}±{int(err)}", ha='center', fontsize=8)

       ax2.set_xticks(x + 1.5 * bar_width)
       ax2.set_xticklabels(star_order)
       ax2.set_ylabel('Detected Planets')
       ax2.set_title(f'Detected Planets by Star Type for {name} ({nruns} Runs)\nStar Catalog: {star_catalog}')
       ax2.legend(title='Radius Bin')
       plt.tight_layout()

       plt.tight_layout()
       data_dir = make_output_dir(name, nruns, star_catalog)

       plt.savefig(os.path.join(data_dir, f"stellar_type_{name}_nruns{nruns}_{star_catalog}.png"), 
                             dpi=300, bbox_inches='tight')

    
def plot_by_planet(df, nruns=1, star_catalog='Gaia', name='hwo'):
       '''
       Produces two plots:
       1. Stacked bar plot: detected vs. undetected planets by category and temperature zone with error bars.
       2. Detected-only bar plot with error bars.
       '''
       df['temp_zone'] = df['temp_p'].apply(temp_zone)
       df['category'] = df.apply(assign_category, axis=1)
       df = df.dropna(subset=['category'])
       df['detected_flag'] = df['detected'].astype(bool)

       total_stats = pivot_stats_temp(df)
       detected_stats = pivot_stats_temp(df[df['detected_flag']])

       categories = ['Rocky eHZ', 'Exo-Earth Candidates', 'Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
       temp_zones = ['hot', 'warm', 'cold']
       offsets = [-0.2, 0, 0.2]
       x = np.arange(len(categories))
       bar_width = 0.2
       colors = ['red', 'gold', 'blue']
       labels = ['hot', 'warm', 'cold']

       def plot_from_stats(stats, title, ylabel):
              fig, ax = plt.subplots(figsize=(10, 6))
              for i, zone in enumerate(temp_zones):
                     data = stats[stats['temp_zone'] == zone].set_index('category').reindex(categories)
                     counts = data['count'].fillna(0).values
                     errors = data['error'].fillna(0).values
                     ax.bar(x + offsets[i], counts, bar_width, label=labels[i],
                            color=colors[i], edgecolor='black', yerr=errors, capsize=4)
                     for j, (h, err) in enumerate(zip(counts, errors)):
                            ax.text(x[j] + offsets[i], h + 1, f"{int(h)}±{int(err)}", ha='center', fontsize=8)
              ax.set_xticks(x)
              ax.set_xticklabels(categories, rotation=15, ha='right')
              ax.set_ylabel(ylabel)
              ax.set_title(title)
              ax.legend(title='Temp Zone')
              plt.tight_layout()
              return fig

       fig1 = plot_from_stats(total_stats, f'Total Planets by Type and Temp Zone\n{name}, {nruns} Runs — {star_catalog}', "Planet Count")
       fig2 = plot_from_stats(detected_stats, f'Detected Planets by Type and Temp Zone\n{name}, {nruns} Runs — {star_catalog}', "Detected Planet Count")

       return fig1, fig2

    
def plot_distances(df, nruns=1, star_catalog='Gaia', name='hwo'):
       '''
       Produces two bar plots:
       1. Stacked bar plot of detected vs. undetected planets by distance bin with error bars.
       2. Detected-only bar plot with error bars.
       '''
       import os

       bins = [0, 3, 5, 7, 9, 11, 13, 15, np.inf]
       labels = ['< 3', '3 - 5', '5 - 7', '7 - 9', '9 - 11', '11 - 13', '13 - 15', '> 15']
       df['distance_bin'] = pd.cut(df['distance_s'], bins=bins, labels=labels, right=False)
       df['detected_flag'] = df['detected'].astype(bool)

       # Compute total and detected counts by run
       def run_bin_counts(subset):
              return subset.groupby(['run', 'distance_bin']).size().unstack(fill_value=0).reindex(columns=labels, fill_value=0)

       total_per_run = run_bin_counts(df)
       detected_per_run = run_bin_counts(df[df['detected_flag']])

       total_mean = total_per_run.mean()
       total_std = total_per_run.std()

       detected_mean = detected_per_run.mean()
       detected_std = detected_per_run.std()

       undetected_mean = total_mean - detected_mean

       x = np.arange(len(labels))
       bar_width = 0.6

       # === Full stacked bar plot with error bars ===
       fig1, ax1 = plt.subplots(figsize=(10, 6))
       ax1.bar(x, undetected_mean, width=bar_width, label='Not detected',
              color='lightgray', edgecolor='black', alpha=0.5)
       ax1.bar(x, detected_mean, width=bar_width, bottom=undetected_mean,
              label='Detected', color='seagreen', edgecolor='black')

       for i, (det, undet, err) in enumerate(zip(detected_mean, undetected_mean, total_std)):
              total = det + undet
              ax1.text(i, total + 2, f"{int(total)}±{int(err)}", ha='center', va='bottom', fontsize=10)

       ax1.set_ylabel("Planet Count")
       ax1.set_xlabel("Distance [pc]")
       ax1.set_xticks(x)
       ax1.set_xticklabels(labels)
       ax1.set_ylim(0, (total_mean + total_std).max() * 1.2)
       ax1.set_title(f'Planet Detection by Distance Bin\n{name}, {nruns} Runs — {star_catalog}')
       ax1.legend()
       plt.tight_layout()

       # === Detected-only bar plot ===
       fig2, ax2 = plt.subplots(figsize=(10, 6))
       ax2.bar(x, detected_mean, width=bar_width, yerr=detected_std,
              capsize=4, color='seagreen', edgecolor='black', label='Detected')

       for i, (val, err) in enumerate(zip(detected_mean, detected_std)):
              ax2.text(i, val + 2, f"{int(val)}±{int(err)}", ha='center', va='bottom', fontsize=10)

       ax2.set_ylabel("Detected Planet Count")
       ax2.set_xlabel("Distance [pc]")
       ax2.set_xticks(x)
       ax2.set_xticklabels(labels)
       ax2.set_ylim(0, (detected_mean + detected_std).max() * 1.2)
       ax2.set_title(f'Detected Planets by Distance Bin\n{name}, {nruns} Runs — {star_catalog}')
       ax2.legend()
       plt.tight_layout()

       data_dir = make_output_dir(name, nruns, star_catalog)

       plt.savefig(os.path.join(data_dir, f"planet_distance_{name}_nruns{nruns}_{star_catalog}.png"), 
                                   dpi=300, bbox_inches='tight')

def plot_by_type(df, nruns=1, star_catalog='Gaia', name='hwo'):
    plot_by_planet(df, nruns=nruns, star_catalog=star_catalog, name=name)
    plot_by_star(df, nruns=nruns, star_catalog=star_catalog, name=name)
    plot_distances(df, nruns=nruns, star_catalog=star_catalog, name=name)
    
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pandas as pd
import multiprocessing as mp
from PPop.StarCatalogs import CrossfieldBrightSample, ExoCat_1, LTC_2, LTC_3

from lifesim.core.hwo_data import HWOData
from ppop_generator import PPop
from tools import PPOP_DATA_DIR

NUNIVERSES = 1
NTEST = 100
NRUNS = 10
STAR_CATALOG = 'LTC_3'#ExoCat_1'  # or 'LTC_3'


def run_single(i):
    '''
    Runs a single instance of the PPop simulation and HWO data analysis.
    '''
    print(f"Running simulation {i}...")
    PPopObj = PPop(seed=i) # i guess we'll reinstantiate each run...
    if STAR_CATALOG == 'ExoCat_1':
        PPopObj.StarCatalog = ExoCat_1
    elif STAR_CATALOG == 'LTC_3':
        PPopObj.StarCatalog = LTC_3

    filename = f'test_runs_{i}'
    data_path = os.path.join(PPOP_DATA_DIR, filename)

    df = PPopObj.run_ppop(data_path, ntest=NTEST, nuniverses=NUNIVERSES)
    PPopObj.catalog_from_ppop(data_path, df=df)
    PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.0)
    PPopObj.catalog_remove_distance(stype='M', mode='larger', dist=10.0)

    hwo_data = HWOData(PPopObj.catalog)
    hwo_data.determine_detectable()

    df = hwo_data.catalog
    bins = [0, 1.5, 3.0, 6.0]
    labels = ['<1.5', '1.5–3.0', '3.0–6.0']
    df['radius_bin'] = pd.cut(df['radius_p'], bins=bins, labels=labels, include_lowest=True)

    # Add "Rocky HZ" bin
    rocky_hz = df[(df['habitable'] == True) & (df['radius_p'] < 1.5)].copy()
    rocky_hz['radius_bin'] = 'Rocky HZ'

    # Combine all rows
    df_all = pd.concat([df, rocky_hz], ignore_index=True)

    # Group by star type and radius bin
    grouped_df = df_all.groupby(['stype', 'radius_bin']).size().reset_index(name='count')

    print(f"Simulation {i} complete.")

    return grouped_df

def main(parallel=False):
    start = time.time()
    n_runs = NRUNS

    # Run in parallel
    with mp.Pool(processes=mp.cpu_count()-2) as pool:
        results = pool.map(run_single, range(n_runs))
    # Combine all run results
    df_all = pd.concat(results, keys=range(n_runs)).reset_index(level=0).rename(columns={'level_0': 'run'})

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

    grouped_sum.to_csv(os.path.join(PPOP_DATA_DIR, 'grouped_sum.csv'))

    # Create grouped bar plot
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
    plt.savefig(f"planets_hwo_nruns{NRUNS}_ntests{NTEST}_nuniverse{NUNIVERSES}_{STAR_CATALOG}.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Total time: {time.time() - start:.2f} seconds")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()

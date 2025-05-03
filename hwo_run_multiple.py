import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pandas as pd
import multiprocessing as mp

from lifesim.core.hwo_data import HWOData
from ppop_generator import PPop
from tools import PPOP_DATA_DIR


def run_single(i):
    '''
    Runs a single instance of the PPop simulation and HWO data analysis.
    '''
    PPopObj = PPop() # i guess we'll reinstantiate each run...

    filename = f'test_runs_{i}'
    data_path = os.path.join(PPOP_DATA_DIR, filename)

    df = PPopObj.run_ppop(data_path, ntest=100, nuniverses=1)
    PPopObj.catalog_from_ppop(data_path, df=df)
    PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.0)
    PPopObj.catalog_remove_distance(stype='M', mode='larger', dist=10.0)

    hwo_data = HWOData(PPopObj.catalog)
    hwo_data.determine_detectable()
    grouped_data = hwo_data.organize_data()

    return grouped_data

def main():
    start = time.time()

    indices = list(range(3))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_single, indices)

    # Collect first result for column naming
    df_hab_total = pd.DataFrame([r['hab_values'] for r in results])
    df_hab_total.columns = results[0]['hab_stypes']

    df_false_total = pd.DataFrame([r['false_values'] for r in results])
    df_false_total.columns = results[0]['false_stypes']

    # Build results summary
    df_results = pd.DataFrame(columns=['stypes', 'count_hab', 'error_hab', 'count_unhab', 'error_unhab'])
    for stype in df_hab_total.columns:
        count = np.mean(df_hab_total[stype])
        err = np.std(df_hab_total[stype])
        count_unhab = np.mean(df_false_total[stype])
        err_unhab = np.std(df_false_total[stype])
        df = pd.DataFrame({
            'stypes': [stype],
            'count_hab': [count],
            'error_hab': [err],
            'count_unhab': [count_unhab],
            'error_unhab': [err_unhab]
        })
        df_all = pd.concat([df_results, df], ignore_index=True)

    # Save results
    df_all.to_csv(os.path.join(PPOP_DATA_DIR, 'hwo_results.csv'), index=False)

    # Define bins and labels
    bins = [0, 1.0, 1.5, 2.5, np.inf]
    labels = ['<1.0', '1.0–1.5', '1.5–2.5', '>2.5']
    df_all['radius_bin'] = pd.cut(df_all['planet_radius'], bins=bins, labels=labels, include_lowest=True)

    # Count by star type and radius bin
    grouped = df_all.groupby(['stype', 'radius_bin']).agg(
        count=('count_overall', 'sum')
    ).reset_index()

    grouped['error'] = grouped['count'].apply(lambda x: x**0.5)

    pivot_counts = grouped.pivot(index='stype', columns='radius_bin', values='count').fillna(0)
    pivot_errors = grouped.pivot(index='stype', columns='radius_bin', values='error').fillna(0)

    # Sort by desired star type order
    star_order = ['F', 'G', 'K', 'M']
    pivot_counts = pivot_counts.reindex(star_order)
    pivot_errors = pivot_errors.reindex(star_order)

    # Build plotting groups
    groups = {
        label: (pivot_counts[label].tolist(), pivot_errors[label].tolist())
        for label in labels
    }

    # Plotting
    colors = ['lightblue', 'deepskyblue', 'midnightblue', 'forestgreen']
    hatches = ['...', 'ooo', 'OO', None]
    bar_width = 0.2
    star_types = pivot_counts.index.tolist()
    x = np.arange(len(star_types))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (label, (heights, errors)) in enumerate(groups.items()):
        ax.bar(x + i * bar_width, heights, width=bar_width,
               yerr=errors, label=label,
               color=colors[i], hatch=hatches[i], edgecolor='black')

        # Optional annotations
        for j, (h, err) in enumerate(zip(heights, errors)):
            ax.text(x[j] + i * bar_width, h + 2, f"{int(h)}±{int(err)}", ha='center', fontsize=8)

    ax.set_xticks(x + 1.5 * bar_width)
    ax.set_xticklabels(star_types)
    ax.set_ylabel('Detectable Planets')
    ax.set_title("Detectable Planets by Star Type (D = 2.0 m, Scenario 1)")
    ax.legend(title='Planet Radius')
    plt.tight_layout()
    plt.show()

    print(f"Total time: {time.time() - start:.2f} seconds")

if __name__ == '__main__':
    mp.set_start_method("spawn")  # especially important for macOS/Windows
    main()

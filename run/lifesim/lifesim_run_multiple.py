import os
import lifesim
from run.ppop.ppop_generator import PPop
import multiprocessing as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


from tools.paths import PPOP_DATA_DIR, LIFESIM_DATA_DIR
from PPop.StarCatalogs import CrossfieldBrightSample, ExoCat_1, LTC_2, LTC_3

STAR_CATALOG = 'LTC_3'#ExoCat_1'  # or 'LTC_3'
RUN_PPOP = False

def run_lifesim_single(i):

    # ----- Generate new planet population -----
    PPopObj = PPop(seed=i)

    if STAR_CATALOG == 'ExoCat_1':
        PPopObj.StarCatalog = ExoCat_1
    elif STAR_CATALOG == 'LTC_3':
        PPopObj.StarCatalog = LTC_3

    filename = f'test_runs_lifesim_{i}'
    data_path = os.path.join(PPOP_DATA_DIR, filename)

    df = PPopObj.run_ppop(seed=i, data_path=data_path)
    PPopObj.catalog_from_ppop(data_path, df=df)
    PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.0)
    PPopObj.catalog_remove_distance(stype='M', mode='larger', dist=10.0)

    # ----- Run LIFEsim with this catalog -----
    bus = lifesim.Bus()
    bus.data.options.set_scenario('baseline')
    bus.data.options.set_manual(diameter=4.0)
    bus.data.options.set_manual(output_path=LIFESIM_DATA_DIR)
    bus.data.options.set_manual(output_filename=f'/test_runs_{i}')
    bus.data.catalog_from_ppop(data_path, df=df)

    # ----- Instrument and Modules -----
    instrument = lifesim.Instrument(name='inst', seed=i)
    bus.add_module(instrument)
    bus.add_module(lifesim.TransmissionMap(name='transm'))
    bus.add_module(lifesim.PhotonNoiseExozodi(name='exo'))
    bus.add_module(lifesim.PhotonNoiseLocalzodi(name='local'))
    bus.add_module(lifesim.PhotonNoiseStar(name='star'))
    bus.connect(('inst', 'transm'))
    bus.connect(('inst', 'exo'))
    bus.connect(('inst', 'local'))
    bus.connect(('inst', 'star'))
    bus.connect(('star', 'transm'))

    opt = lifesim.Optimizer(name='opt')
    ahgs = lifesim.AhgsModule(name='ahgs')
    bus.add_module(opt)
    bus.add_module(ahgs)
    bus.connect(('transm', 'opt'))
    bus.connect(('inst', 'opt'))
    bus.connect(('opt', 'ahgs'))

    # ----- Run Simulation -----
    instrument.get_snr()
    opt.ahgs()
    bus.save()

    return bus.data.catalog

def assign_radius_bin(r):
    if r < 1.5:
        return '<1.5'
    elif r < 3.0:
        return '1.5–3.0'
    elif r < 6.0:
        return '3.0–6.0'
    else:
        return 'Rocky HZ'
    
def main(parallel=True, nruns=1):
    start=time.time()
    if parallel:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(run_lifesim_single, range(nruns))
    else:
        results = [run_lifesim_single(i) for i in range(nruns)]

    # Parameters
    nruns = len(results)  # List of DataFrames, one per run
    star_order = ['F', 'G', 'K', 'M']
    bin_labels = ['<1.5', '1.5–3.0', '3.0–6.0', 'Rocky HZ']

    # Step 1: Process each run
    processed_runs = []

    for run_idx, df in enumerate(results):
        df_detected = df[df['detected'] == True].copy()
        df_detected['radius_bin'] = df_detected['radius_p'].apply(assign_radius_bin)

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
    plt.savefig(f"planets_lifesim_nruns{nruns}_{STAR_CATALOG}.png", dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Total time: {time.time() - start:.2f} seconds")

if __name__ == '__main__':
    
    mp.set_start_method('spawn')
    main()


"""Run the quickstart or select a full simulation pipeline under ``__main__``."""

import time
import os
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from tools.paths import LOGGING

from run.kepler.run_kepler import main as main_kepler
from run.tess.run_tess import main as main_tess
from run.rv.run_rv import main as main_rv
from run.hwo.run_hwo import main as main_hwo
from run.lifesim.run_lifesim import main as main_lifesim
from run.flat_universe.run_flat_universe import main as main_flat_ab
from run.multi_run import normalize_nruns

from plotting.plot import plot_all
from plotting.plot_flat_universe import plot_flat_universe

def run_with_progress(func, name, estimated_minutes=12, *args, **kwargs):
    # Logs to file and runs func. Real progress comes from the per-star tqdm bar
    # inside the generator (and the per-universe "Running ... for run i" prints),
    # not from a fake wall-clock estimate.
    log_path = os.path.join(LOGGING, name, "run_log" + datetime.now().strftime("_%Y%m%d_%H%M%S") + ".txt")
    log_dir = os.path.dirname(log_path)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    print(f"Writing log to: {log_path}")
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')
    def log(msg):
        print(msg)
        logging.info(msg)
    try:
        log(f"Starting function '{func.__name__}'...")
        result = func(*args, **kwargs)
        log(f"Function '{func.__name__}' completed successfully.")
    except Exception as e:
        log(f"Error during '{func.__name__}': {e}")
        raise
    return result

def run_sim(func=main_kepler, name='kepler', parallel=True, nruns=500,
            star_catalog='Gaia', run_anew=True, plot=True, **kwargs):
    nruns = normalize_nruns(nruns)
    start_time = time.time()
    print(f"Starting simulation: {name} with {len(nruns)} universe(s)...")
    try:
        df_concat = run_with_progress(
            func,
            name=name,
            estimated_minutes=12,
            parallel=parallel,
            nruns=nruns,
            star_catalog=star_catalog,
            run_anew=run_anew,
            **kwargs
        )
    finally:
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"\nSimulation completed in: {hours}:{minutes:02d}:{seconds:02d}")
    if 'radius_bin' not in df_concat.columns:
        bins = [0, 1.5, 3.0, 6.0]
        labels = ['<1.5', '1.5–3.0', '3.0–6.0']
        df_concat['radius_bin'] = pd.cut(df_concat['radius_p'], bins=bins, labels=labels, include_lowest=True)

    if plot:
        plot_start_time = time.time()
        print(f"Starting plotting...")

        # Universes A and B have their own plotter
        if name.lower() == 'flat_ab':
            plot_flat_universe(df=df_concat, nruns=len(nruns), use_multiprocessing=False)
        else:
            plot_all(df=df_concat, sim_name=name, nruns=len(nruns), star_catalog=star_catalog, use_multiprocessing=False)

        plot_end_time = time.time()
        plot_elapsed = plot_end_time - plot_start_time
        plot_hours = int(plot_elapsed // 3600)
        plot_minutes = int((plot_elapsed % 3600) // 60)
        plot_seconds = int(plot_elapsed % 60)
        print(f"Time taken to plot: {plot_hours}:{plot_minutes:02d}:{plot_seconds:02d}")
    return df_concat


def run_flat_nonphysical():
    """Flat-population analysis used for a paper figure (not the quickstart)."""
    print("\nRunning flat_nonphysical: rocky planets around G, K and M stars through the "
          "TESS transit and HARPS/NIRPS RV detection models.\n")
    # Imported here so the P-Pop runs skip the paper-figure code.
    from plotting.scripts.analysis import flat_transit_rv_3x3
    flat_transit_rv_3x3.main(paper_copy=False)


# The guard is required: parallel runs start worker processes that re-import this file.
if __name__ == "__main__":

    # Universe i uses seed i. The Gaia-60pc catalog (~54k stars, ~73% M dwarfs) gives ~100k
    # planets per universe, so the diagnostics stack 10 to fill the sparse F/G/K bins.
    NRUNS = np.arange(1)
    STAR_CATALOG = 'Gaia'  # or 'ExoCat_1', 'LTC_2', 'LTC_3', 'CrossfieldBrightSample'

    # Quickstart: flat universes A and B through Kepler, TESS and RV detection.
    run_sim(func=main_flat_ab, name='flat_ab', parallel=False, nruns=np.arange(1),
            run_anew=True, plot=True, seed=0, n_planets=20_000)

    # Optional P-Pop quickstart: comment out the flat run above and uncomment this line.
    # run_sim(func=main_kepler, name='kepler_ppop', parallel=False, nruns=np.arange(1), star_catalog='LTC_2', run_anew=True, plot=True)

    # Full P-Pop diagnostics: set NRUNS = np.arange(10), comment out the quickstart,
    # and uncomment both lines. Each Gaia-60pc universe takes about 20 minutes; 10 Kepler
    # plus 10 TESS universes take roughly 3-4 hours and feed the optional --full analyses.
    # run_sim(func=main_kepler, name='kepler', parallel=True, nruns=NRUNS, star_catalog=STAR_CATALOG, run_anew=True, plot=True)
    # run_sim(func=main_tess, name='TESS', parallel=True, nruns=NRUNS, star_catalog=STAR_CATALOG, run_anew=True, plot=True)

    # Other pipelines:
    # run_sim(func=main_rv, name='rv', parallel=True, nruns=NRUNS, star_catalog=STAR_CATALOG, run_anew=True, plot=True)
    # run_sim(func=main_hwo, name='hwo', parallel=False, nruns=NRUNS, star_catalog=STAR_CATALOG, run_anew=True, plot=True)
    # run_sim(func=main_lifesim, name='lifesim', parallel=False, nruns=NRUNS, star_catalog=STAR_CATALOG, run_anew=True, plot=True)
    # run_sim(func=main_flat_ab, name='flat_ab', parallel=False, nruns=np.arange(1), run_anew=True, plot=True)  # universes A and B, ~1 min

    print('done')

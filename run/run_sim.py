import time
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from tqdm import tqdm
import threading

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.paths import LOGGING, LIFESIM_OUTER_DIR
from lifesim.core.hwo_data import HWOData
from run.lifesim.lifesim_run_multiple import main as main_lifesim
from run.hwo.hwo_run_multiple import main as main_hwo
from lifesim.core.hwo_data import HWOData
from plot.plot import plot_all
from tools.exoplanet_catalog import load_and_filter_exoplanets

def run_with_progress(func, name, estimated_minutes=12, *args, **kwargs):
    estimated_seconds = estimated_minutes * 60
    log_path = os.path.join(LOGGING, name, "run_log" + datetime.now().strftime("_%Y%m%d_%H%M%S") + ".txt")
    log_dir = os.path.dirname(log_path)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    print(f"Writing log to: {log_path}")
    logging.basicConfig(filename=log_path, filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s')
    def log(msg):
        print(msg)
        logging.info(msg)
    stop_event = threading.Event()
    def time_based_progress_bar(estimated_seconds, stop_event, log_func=None):
        with tqdm(total=estimated_seconds, unit='s', ncols=80) as pbar:
            start_time = time.time()
            while not stop_event.is_set():
                elapsed = time.time() - start_time
                if elapsed >= estimated_seconds:
                    break
                pbar.n = int(elapsed)
                pbar.refresh()
                if log_func:
                    log_func(f"Progress: {int(elapsed)} / {estimated_seconds} seconds")
                time.sleep(1)
            pbar.n = estimated_seconds
            pbar.refresh()
            if log_func:
                log_func("Progress complete.")
    progress_thread = threading.Thread(target=time_based_progress_bar, args=(estimated_seconds, stop_event, log))
    progress_thread.start()
    try:
        log(f"Starting function '{func.__name__}'...")
        result = func(*args, **kwargs)
        log(f"Function '{func.__name__}' completed successfully.")
    except Exception as e:
        log(f"Error during '{func.__name__}': {e}")
        raise
    finally:
        stop_event.set()
        progress_thread.join()
        log("Progress thread joined. Wrapper complete.")
    return result

def run_sim(func=main_hwo, name='hwo', parallel=True, nruns=500, star_catalog='Gaia', run_anew=True, plot=True):
    start_time = time.time()
    print(f"Starting simulation: {name} with {len(nruns)} runs...")
    print("Elapsed time: 0:00:00", end='', flush=True)
    def update_timer():
        while True:
            elapsed = time.time() - start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            print(f"\rElapsed time: {hours}:{minutes:02d}:{seconds:02d}", end='', flush=True)
            time.sleep(1)
    timer_thread = threading.Thread(target=update_timer, daemon=True)
    timer_thread.start()
    try:
        df_concat = run_with_progress(
            func,
            name=name,
            estimated_minutes=12,
            parallel=parallel,
            nruns=nruns,
            star_catalog=star_catalog,
            run_anew=run_anew
        )
    finally:
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        print(f"\rSimulation completed in: {hours}:{minutes:02d}:{seconds:02d}")
    bins = [0, 1.5, 3.0, 6.0]
    labels = ['<1.5', '1.5–3.0', '3.0–6.0']
    df_concat['radius_bin'] = pd.cut(df_concat['radius_p'], bins=bins, labels=labels, include_lowest=True)
    if plot:
        plot_start_time = time.time()
        print(f"Starting plotting...")
        plot_all(df=df_concat, sim_name=name, nruns=len(nruns), star_catalog=star_catalog, use_multiprocessing=False)
        plot_end_time = time.time()
        plot_elapsed = plot_end_time - plot_start_time
        plot_hours = int(plot_elapsed // 3600)
        plot_minutes = int((plot_elapsed % 3600) // 60)
        plot_seconds = int(plot_elapsed % 60)
        print(f"Time taken to plot: {plot_hours}:{plot_minutes:02d}:{plot_seconds:02d}")
    return df_concat

def run_exoplanet_plotting(name='HWO_exoplanets', star_catalog='exoplanet_catalog', plot=True):
    print(f"Loading exoplanets data for plotting...")
    exo_path = os.path.join(LIFESIM_OUTER_DIR, 'exoplanets_2025.csv')
    df = load_and_filter_exoplanets(exo_path, instrument='HWO')
    
    # Process through HWO detection simulation
    print(f"Processing {len(df)} planets through HWO detection simulation...")
    hwo_data = HWOData(df)
    hwo_data.determine_detectable()
    
    # Get the processed data with detection results
    df = hwo_data.catalog
    print(f"After HWO detection processing: {len(df)} planets")
    print(f"Detected planets (best case): {df['detected_best'].sum() if 'detected_best' in df.columns else 'N/A'}")
    print(f"Detected planets (worst case): {df['detected_worst'].sum() if 'detected_worst' in df.columns else 'N/A'}")

    # DEBUG: Check detected_best value counts right before plotting
    print('DEBUG: Value counts for detected_best before plotting:')
    print(df['detected_best'].value_counts())
    print('Sample detected_best:', df['detected_best'].head(10).tolist())
    
    required_columns = [
        'luminosity_s', 'distance_s', 'radius_p', 'p_orb', 'semimajor_p', 'temp_s', 'temp_p', 'mass_p',
        'detected', 'detected_best', 'detected_worst', 'flux_ratio_value_best', 'maxangsep', 'z',
        'stype', 'habitable', 'run', 'radius_bin'
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for plotting: {missing}")

    print(f"Loaded {len(df)} exoplanet records")

    if plot:
        plot_start_time = time.time()
        print(f"Starting plotting for exoplanets data...")
        plot_all(df=df, sim_name=name, nruns=1, star_catalog=star_catalog, use_multiprocessing=False)
        plot_end_time = time.time()
        plot_elapsed = plot_end_time - plot_start_time
        plot_hours = int(plot_elapsed // 3600)
        plot_minutes = int((plot_elapsed % 3600) // 60)
        plot_seconds = int(plot_elapsed % 60)
        print(f"Time taken to plot: {plot_hours}:{plot_minutes:02d}:{plot_seconds:02d}")

    return df


# Run the whole thing
if __name__ == "__main__":

    NRUNS = np.arange(1)
    
    # Run exoplanet plotting
    # run_exoplanet_plotting(name='HWO_exoplanets', star_catalog='Gaia', plot=True)
    
    run_sim(func=main_hwo, name = 'hwo', parallel=False, nruns=NRUNS, star_catalog='Gaia', run_anew=False)
    # # print('Completed HWO')
    # run_sim(func=main_lifesim, name='lifesim', parallel=True, nruns=NRUNS, star_catalog='Gaia', run_anew=False)
    # # print('Completed Lifesim Gaia')
    # run_sim(func=main_lifesim, name='lifesim', parallel=True, nruns=NRUNS, star_catalog='LTC_3', run_anew=False)
    # print('Completed Lifesim LTC_3')
    print('done')
import time
import threading
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
import logging
import numpy as np
import pandas as pd
from datetime import datetime

from tools.paths import LOGGING
from run.lifesim.lifesim_run_multiple import main as main_lifesim
from run.hwo.hwo_run_multiple import main as main_hwo
from plot.plot import plot_all


def time_based_progress_bar(estimated_seconds, stop_event, log_func=None):
    '''
    Implements a time-based progress bar that runs for a specified number of seconds.
    '''
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

def run_with_progress(func, name, estimated_minutes=12, *args, **kwargs):
    '''
    Runs simulation with a progress bar and logging.
    '''
    estimated_seconds = estimated_minutes * 60

    log_path=os.path.join(LOGGING, name, "run_log"+ datetime.now().strftime("_%Y%m%d_%H%M%S") + ".txt")

    # Set up logger
    logging.basicConfig(filename=log_path,
                        filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s')

    def log(msg):
        print(msg)
        logging.info(msg)

    stop_event = threading.Event()
    progress_thread = threading.Thread(target=time_based_progress_bar,
                                       args=(estimated_seconds, stop_event, log))
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

def run_sim(func, name, parallel, nruns, star_catalog, run_anew=True):
    '''
    Runs the simulation with the provided function, name, parallel execution flag,
    number of runs, and star catalog.'''
    df_concat = run_with_progress(
        func,
        name=name,
        estimated_minutes=12,
        parallel=parallel,
        nruns=nruns,
        star_catalog=star_catalog,
        run_anew=run_anew
    )

    # Bin by radius
    bins = [0, 1.5, 3.0, 6.0]
    labels = ['<1.5', '1.5–3.0', '3.0–6.0']
    df_concat['radius_bin'] = pd.cut(df_concat['radius_p'], bins=bins, labels=labels, include_lowest=True)

    # Add "Rocky HZ" bin
    rocky_hz = df_concat[(df_concat['habitable'] == True) & (df_concat['radius_p'] < 1.5)].copy()
    rocky_hz['radius_bin'] = 'Rocky HZ'

    # Combine all rows
    df_concat = pd.concat([df_concat, rocky_hz], ignore_index=True)

    start_time = time.time()
    plot_all(df=df_concat, sim_name=name, nruns=len(nruns), star_catalog=star_catalog)
    end_time = time.time()
    print(f"Time taken to plot: {end_time - start_time} seconds")

# Run the whole thing
if __name__ == "__main__":

    PARALLEL = True  # Set to True if you want to run in parallel
    STAR_CATALOG = 'Gaia'  # or 'ExoCat_1'
    NRUNS = np.arange(500)
    
    run_sim(func=main_hwo, name = 'hwo', parallel=PARALLEL, nruns=NRUNS, star_catalog=STAR_CATALOG, run_anew=False)
    # run_sim(func=main_lifesim, name='lifesim',parallel=PARALLEL, nruns=NRUNS, star_catalog=STAR_CATALOG, run_anew=False)
    # run_sim(func=main_lifesim, name='lifesim', parallel=PARALLEL, nruns=NRUNS, star_catalog='LTC_3', run_anew=False)
    print('done')
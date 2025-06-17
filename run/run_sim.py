import time
import threading
import os
from tqdm import tqdm
from datetime import datetime
from tools.paths import LOGGING
from run.lifesim.lifesim_run_multiple import main as main_lifesim
from run.hwo.hwo_run_multiple import main as main_hwo

import time
import threading
import logging
from tqdm import tqdm

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

def run_with_progress(func, sim_name, estimated_minutes=12, *args, **kwargs):
    estimated_seconds = estimated_minutes * 60

    log_path=os.path.join(LOGGING, sim_name, "run_log"+ datetime.now().strftime("_%Y%m%d_%H%M%S") + ".txt")

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

# Run the whole thing
if __name__ == "__main__":
    NRUNS = 2
    PARALLEL = True  # Set to True if you want to run in parallel
    STAR_CATALOG = 'Gaia'  # or 'ExoCat_1'
    SIM = 'hwo'
    if SIM == 'lifesim':
        main = main_lifesim
    elif SIM == 'hwo':
        main = main_hwo
    else:
        raise ValueError("Invalid simulation type. Choose 'lifesim' or 'hwo'.")
    run_with_progress(func=main, sim_name = SIM, parallel=PARALLEL, nruns=NRUNS, star_catalog=STAR_CATALOG)
import time
import threading
from tqdm import tqdm
from run.lifesim.lifesim_run_multiple import main

def time_based_progress_bar(estimated_seconds, stop_event):
    with tqdm(total=estimated_seconds, unit='s', ncols=80) as pbar:
        start_time = time.time()

        while not stop_event.is_set():
            elapsed = time.time() - start_time
            if elapsed >= estimated_seconds:
                break
            pbar.n = int(elapsed)
            pbar.refresh()
            time.sleep(0.5)

        # Finish bar cleanly
        pbar.n = estimated_seconds
        pbar.refresh()

def run_with_progress(parallel, nruns=1, star_catalog='Gaia'):
    estimated_minutes = 12
    estimated_seconds = estimated_minutes * 60

    # Thread stop signal
    stop_event = threading.Event()

    # Start the progress bar in a background thread
    progress_thread = threading.Thread(target=time_based_progress_bar, args=(estimated_seconds, stop_event))
    progress_thread.start()

    try:
        # Run your actual function here
        main(parallel= parallel,nruns=nruns, star_catalog=star_catalog)

    finally:
        # Signal the progress thread to stop and wait for it
        stop_event.set()
        progress_thread.join()

    print("Function completed.")

# Run the whole thing
if __name__ == "__main__":
    NRUNS = 2
    PARALLEL = True  # Set to True if you want to run in parallel
    STAR_CATALOG = 'Gaia'  # or 'ExoCat_1'
    run_with_progress(parallel=PARALLEL, nruns=NRUNS, star_catalog=STAR_CATALOG)

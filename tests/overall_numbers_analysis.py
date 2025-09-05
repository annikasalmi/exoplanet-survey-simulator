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

from run.hwo.hwo_run_multiple import main as main_hwo

# Define LOGGING directory
LOGGING = os.path.join(os.path.dirname(__file__), '..', 'logs')

def run_with_progress(func, name, estimated_minutes=12, *args, **kwargs):
    estimated_seconds = estimated_minutes * 60
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.join(LOGGING, name), exist_ok=True)
    log_path = os.path.join(LOGGING, name, "run_log" + datetime.now().strftime("_%Y%m%d_%H%M%S") + ".txt")
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

def run_sim(func=main_hwo, name='hwo', parallel=True, nruns=np.arange(500), star_catalog='Gaia', run_anew=True, plot=True):
    start_time = time.time()
    # Fix the len() issue by checking if nruns is iterable
    if isinstance(nruns, (list, tuple, np.ndarray)):
        nruns_count = len(nruns)
    else:
        nruns_count = nruns
    print(f"Starting simulation: {name} with {nruns_count} runs...")
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

    # Plot the data
    return df_concat

def get_detection_stats(df_concat):
    """
    Analyze detection statistics per run across all 500 runs.
    
    Args:
        df_concat: DataFrame containing results from multiple runs
        
    Returns:
        dict: Dictionary containing various statistics per run
    """
    # Group by run number to get per-run statistics
    if 'run' not in df_concat.columns:
        print("Warning: 'run' column not found. Assuming all data is from a single run.")
        run_stats = {
            'detected_planets': df_concat['detected_best'].sum(),
            'total_planets': len(df_concat),
            'detection_rate': df_concat['detected_best'].mean(),
            'missed_planets': (~df_concat['detected_best']).sum()
        }
    else:
        # Group by run number and calculate statistics per run
        run_stats = df_concat.groupby('run').agg({
            'detected_best': ['sum', 'count', 'mean']
        }).round(4)
        
        # Flatten column names
        run_stats.columns = ['detected_planets', 'total_planets', 'detection_rate']
        
        # Add additional statistics
        run_stats['missed_planets'] = run_stats['total_planets'] - run_stats['detected_planets']
        
        # Calculate overall statistics across all runs
        n_runs = len(run_stats)
        mean_detection_rate = run_stats['detection_rate'].mean()
        std_detection_rate = run_stats['detection_rate'].std()
        
        # Calculate standard error of the mean (SEM)
        sem_detection_rate = std_detection_rate / np.sqrt(n_runs)
        
        # Calculate confidence intervals (95% CI)
        from scipy import stats
        confidence_level = 0.95
        t_value = stats.t.ppf((1 + confidence_level) / 2, df=n_runs-1)
        margin_of_error = t_value * sem_detection_rate
        
        overall_stats = {
            'mean_detection_rate': mean_detection_rate,
            'std_detection_rate': std_detection_rate,
            'sem_detection_rate': sem_detection_rate,
            'margin_of_error_95ci': margin_of_error,
            'ci_lower_95': mean_detection_rate - margin_of_error,
            'ci_upper_95': mean_detection_rate + margin_of_error,
            'median_detection_rate': run_stats['detection_rate'].median(),
            'min_detection_rate': run_stats['detection_rate'].min(),
            'max_detection_rate': run_stats['detection_rate'].max(),
            'total_runs': n_runs,
            'runs_with_detections': (run_stats['detected_planets'] > 0).sum(),
            'runs_without_detections': (run_stats['detected_planets'] == 0).sum()
        }
        
        run_stats['overall_stats'] = overall_stats
    
    return run_stats


# Run the whole thing
if __name__ == "__main__":

    NRUNS = np.arange(500)  # Changed back to array since main_hwo expects iterable
    
    print("Starting analysis...")
    try:
        df = run_sim(func=main_hwo, name='hwo', parallel=True, nruns=NRUNS, star_catalog='Gaia', run_anew=False)
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {list(df.columns)}")
        
        detection_stats = get_detection_stats(df)
        print(f"Detection stats type: {type(detection_stats)}")
        
        # Print summary statistics
        print("\n=== Detection Statistics Summary ===")
        if isinstance(detection_stats, pd.DataFrame):
            # Per-run stats: print summary
            mean = detection_stats['detection_rate'].mean()
            std = detection_stats['detection_rate'].std()
            sem = std / np.sqrt(len(detection_stats))
            from scipy import stats
            confidence_level = 0.95
            t_value = stats.t.ppf((1 + confidence_level) / 2, df=len(detection_stats)-1)
            margin_of_error = t_value * sem
            print(f"Mean detection rate: {mean:.4f} ± {sem:.4f}")
            print(f"95% Confidence Interval: [{mean - margin_of_error:.4f}, {mean + margin_of_error:.4f}]")
            print(f"Standard deviation: {std:.4f}")
            print(f"Standard error of mean: {sem:.4f}")
            print(f"Margin of error (95% CI): ±{margin_of_error:.4f}")
            print(f"Median detection rate: {detection_stats['detection_rate'].median():.4f}")
            print(f"Range: [{detection_stats['detection_rate'].min():.4f}, {detection_stats['detection_rate'].max():.4f}]")
            print(f"Runs with detections: {(detection_stats['detected_planets'] > 0).sum()}/{len(detection_stats)}")
            print(f"Runs without detections: {(detection_stats['detected_planets'] == 0).sum()}/{len(detection_stats)}")
            print(f"Total simulations: {len(detection_stats)}")
            print("\nSample per-run stats:")
            print(detection_stats.head())
        else:
            print(f"Detection rate: {detection_stats['detection_rate']:.4f}")
            print(f"Detected planets: {detection_stats['detected_planets']}")
            print(f"Total planets: {detection_stats['total_planets']}")
            print(f"Missed planets: {detection_stats['missed_planets']}")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
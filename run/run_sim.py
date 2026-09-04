import time
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tools.paths import LOGGING, LIFESIM_OUTER_DIR
from telescopes.hwo.detection_model import HWOData
from telescopes.kepler.detection_model import KeplerData #added by Hongyi
from telescopes.tess.detection_model import TESSData


from run.lifesim.lifesim_run_multiple import main as main_lifesim
from run.hwo.hwo_run_multiple import main as main_hwo
from run.kepler.run_kepler import main as main_kepler
from run.tess.run_tess import main as main_tess
from run.flat_universe.run_flat_universe import main as main_flat_universe

from output.plots.plot import plot_all
from output.plots.plot_flat_universe import plot_flat_universe
from tools.exoplanet_catalog import load_and_filter_exoplanets

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

def run_sim(func=main_hwo, name='hwo', parallel=True, nruns=500, star_catalog='Gaia', run_anew=True, plot=True, **kwargs):
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

        # Use dedicated plotter for flat_universe
        if name.lower() == 'flat_universe':
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


def run_exoplanet_plotting(name='HWO_exoplanets', star_catalog='exoplanet_catalog', plot=True):
    print("Loading exoplanets data for plotting...")

    exo_path = os.path.join(LIFESIM_OUTER_DIR, 'exoplanets_2026.csv')
    name_lower = name.lower()

    telescope_map = {
        "hwo": ("HWO", HWOData),
        "kepler": ("Kepler", KeplerData),
        "tess": ("TESS", TESSData),
    }

    telescope_name, DataClass = None, None
    for key, value in telescope_map.items():
        if key in name_lower:
            telescope_name, DataClass = value
            break

    if telescope_name is None:
        raise ValueError(
            f"Unknown telescope name: {name}. "
            "Use 'HWO_exoplanets', 'Kepler_exoplanets', or later 'TESS_exoplanets'."
        )

    # Load catalog for the chosen telescope
    df = load_and_filter_exoplanets(exo_path, instrument=telescope_name)

    print(f"Processing {len(df)} planets through {telescope_name} detection simulation...")

    # Run telescope model
    telescope_data = DataClass(df)
    telescope_data.determine_detectable()
    df = telescope_data.catalog

    # Shared plotting columns
    df["run"] = 0
    df["radius_bin"] = pd.cut(
        df["radius_p"],
        bins=[0, 1.5, 3.0, 6.0],
        labels=["<1.5", "1.5–3.0", "3.0–6.0"],
        include_lowest=True
    )

    # Standardize detection columns
    if "detected" not in df.columns and "detected_best" in df.columns:
        df["detected"] = df["detected_best"]

    if "detected_best" not in df.columns and "detected" in df.columns:
        df["detected_best"] = df["detected"]

    if "detected_worst" not in df.columns and "detected_best" in df.columns:
        df["detected_worst"] = df["detected_best"]

    # Compact debug summary
    print(f"After {telescope_name}: {len(df)} planets")
    for col in ["detected", "detected_best", "detected_worst"]:
        if col in df.columns:
            print(f"{col}: {df[col].sum()} detected")
            print(df[col].value_counts(dropna=False))

    # Required columns
    common_required = [
        "luminosity_s", "distance_s", "radius_p", "p_orb", "semimajor_p",
        "temp_s", "temp_p", "mass_p", "detected", "detected_best",
        "detected_worst", "stype", "habitable", "run", "radius_bin"
    ]

    telescope_required = {
        "HWO": ["flux_ratio_value_best", "maxangsep", "z"],
        "Kepler": [
            "transiting_geometric",
            "transit_depth_ppm",
            "kepler_star_bright_enough",
            "kepler_depth_pass",
        ],
        "TESS": [
            "tess_transiting_geometric",
            "tess_transit_depth_ppm",
            "tess_star_bright_enough",
            "tess_depth_pass",
            "tess_snr",
            "tess_n_transits",
            "tess_reason_category",
        ],
    }

    required_columns = common_required + telescope_required.get(telescope_name, [])
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns for {telescope_name}: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    if plot:
        plot_start_time = time.time()
        print(f"Starting plotting for {telescope_name}...")

        plot_all(
            df=df,
            sim_name=name,
            nruns=1,
            star_catalog=star_catalog,
            use_multiprocessing=False,
        )

        plot_elapsed = time.time() - plot_start_time
        h, rem = divmod(int(plot_elapsed), 3600)
        m, s = divmod(rem, 60)
        print(f"Time taken to plot: {h}:{m:02d}:{s:02d}")

    return df


# Run the whole thing
if __name__ == "__main__":

    # The Gaia-60pc catalog holds ~54k stars, so ONE universe already yields
    # ~100k planets. The local volume is ~73% M dwarfs, so F/G/K panels are
    # sparse from a single universe. Stacking N seeded universes (same fixed
    # star catalog, different RNG seed = independent Monte-Carlo realizations)
    # multiplies planets-per-bin by N and smooths the FGK detection background.
    # 10 universes on 5 workers = 2 batches (~10 h overnight on this box).
    NRUNS = np.arange(10)

    # Run exoplanet plotting
    # run_exoplanet_plotting(name='HWO_exoplanets', star_catalog='Gaia', plot=True)

    run_sim(func=main_kepler, name = 'kepler', parallel=True, nruns=NRUNS, star_catalog='Gaia', run_anew=True, plot=True) # Gaia 60pc
    print('Completed Lifesim kepler')

    run_sim(func=main_tess, name = 'TESS', parallel=True, nruns=NRUNS, star_catalog='Gaia', run_anew=True, plot=True) # Gaia 60pc
    print('Completed Lifesim tess')
    
    # run_sim(func=main_hwo, name = 'hwo', parallel=False, nruns=NRUNS, star_catalog='Gaia', run_anew=True) # CHANGED run_anew to TRUE to re-run HWO with Gaia catalog
    # # print('Completed HWO')
    
    # run_sim(func=main_lifesim, name='lifesim', parallel=False, nruns=NRUNS, star_catalog='Gaia', run_anew=True, plot=True)
    # print('Completed Lifesim Gaia')

    # run_sim(func=main_lifesim, name='lifesim', parallel=True, nruns=NRUNS, star_catalog='LTC_3', run_anew=False)
    # print('Completed Lifesim LTC_3')
    print('done')
import numpy as np
from run.lifesim.lifesim_run_multiple import main as main_lifesim
from run.hwo.hwo_run_multiple import main as main_hwo
from run.kepler.run_kepler import main as main_kepler
from run.tess.run_tess import main as main_tess
from output.plots.plot import plot_all

# --- DEMO SCRIPT FOR NEW USERS ---
# This script runs a basic exoplanet detection simulation and generates plots.
# No user input is required. You can change the parameters below if desired.

# Sensible defaults
NRUNS = 10
STAR_CATALOG = 'Gaia'  # or 'ExoCat_1'
SIM_NAME = 'kepler'    # options: 'hwo', 'lifesim', 'kepler', 'tess'

print("\nWelcome to the exoplanet-survey-simulator demo!\n")
print(f"Running {SIM_NAME} simulation with {NRUNS} runs and catalog '{STAR_CATALOG}'...\n")

# Select simulation function
sim_funcs = {
    'hwo': main_hwo,
    'lifesim': main_lifesim,
    'kepler': main_kepler,
    'tess': main_tess,
}

if SIM_NAME not in sim_funcs:
    print(f"Error: unknown simulation '{SIM_NAME}'. Options: {list(sim_funcs.keys())}")
    exit(1)

default_sim_func = sim_funcs[SIM_NAME]

try:
    df_concat = default_sim_func(parallel=True, nruns=np.arange(NRUNS), star_catalog=STAR_CATALOG, run_anew=True)
    print("Simulation complete! Generating plots...\n")
    plot_all(df=df_concat, sim_name=SIM_NAME, nruns=NRUNS, star_catalog=STAR_CATALOG, use_multiprocessing=True)
    print("All done! Check the output plots in the 'output' directory.")
except Exception as e:
    print(f"\nAn error occurred during the demo run: {e}\n")
    print("Please check your installation and try again, or contact the project maintainers for help.") 
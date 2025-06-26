import numpy as np
from run.lifesim.lifesim_run_multiple import main as main_lifesim
from run.hwo.hwo_run_multiple import main as main_hwo
from plot.plot import plot_all

# --- DEMO SCRIPT FOR NEW USERS ---
# This script runs a basic exoplanet detection simulation and generates plots.
# No user input is required. You can change the parameters below if desired.

# Sensible defaults
NRUNS = 100
STAR_CATALOG = 'Gaia'  # or 'ExoCat_1'
SIM_NAME = 'hwo'       # or 'lifesim'

print("\nWelcome to the mdwarf-habitability demo!\n")
print(f"Running {SIM_NAME} simulation with {NRUNS} runs and catalog '{STAR_CATALOG}'...\n")

# Select simulation function
default_sim_func = main_hwo if SIM_NAME == 'hwo' else main_lifesim

try:
    df_concat = default_sim_func(parallel=True, nruns=np.arange(NRUNS), star_catalog=STAR_CATALOG, run_anew=True)
    print("Simulation complete! Generating plots...\n")
    plot_all(df=df_concat, sim_name=SIM_NAME, nruns=NRUNS, star_catalog=STAR_CATALOG, use_multiprocessing=True)
    print("All done! Check the output plots in the 'plot' directory.")
except Exception as e:
    print(f"\nAn error occurred during the demo run: {e}\n")
    print("Please check your installation and try again, or contact the project maintainers for help.") 
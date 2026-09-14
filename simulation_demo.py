import numpy as np

from run.lifesim.lifesim_run_multiple import main as main_lifesim
from run.hwo.hwo_run_multiple import main as main_hwo
from run.kepler.run_kepler import main as main_kepler
from run.tess.run_tess import main as main_tess
from run.rv.run_rv import main as main_rv
from run.flat_universe.run_flat_universe import main as main_flat_universe
from plotting.plot import plot_all
from plotting.plot_flat_universe import plot_flat_universe

# --- DEMO SCRIPT FOR NEW USERS ---
# Runs one detection simulation and plots the result. No input needed.

NRUNS = 1
STAR_CATALOG = 'Gaia'        # or 'ExoCat_1'
SIM_NAME = 'flat_universe'   # 'flat_universe', 'kepler', 'tess', 'rv', 'hwo', 'lifesim'

# flat_universe draws its own stars and takes under a minute. The others build a
# P-Pop universe per run, which takes about 30 minutes each.
sim_funcs = {
    'flat_universe': main_flat_universe,
    'hwo': main_hwo,
    'lifesim': main_lifesim,
    'kepler': main_kepler,
    'tess': main_tess,
    'rv': main_rv,
}

if SIM_NAME not in sim_funcs:
    raise SystemExit(f"Unknown simulation {SIM_NAME!r}. Options: {list(sim_funcs)}")

print(f"\nRunning {SIM_NAME} with {NRUNS} run(s), catalog {STAR_CATALOG!r}.\n")

df = sim_funcs[SIM_NAME](parallel=True, nruns=np.arange(NRUNS),
                         star_catalog=STAR_CATALOG, run_anew=True)

print("Simulation done. Plotting.\n")
if SIM_NAME == 'flat_universe':
    plot_flat_universe(df, nruns=NRUNS)
else:
    plot_all(df=df, sim_name=SIM_NAME, nruns=NRUNS,
             star_catalog=STAR_CATALOG, use_multiprocessing=True)

print("Done. Plots are under results/.")

import numpy as np

from run.lifesim.lifesim_run_multiple import main as main_lifesim
from run.hwo.hwo_run_multiple import main as main_hwo
from run.kepler.run_kepler import main as main_kepler
from run.tess.run_tess import main as main_tess
from run.rv.run_rv import main as main_rv
from plotting.plot import plot_all

# --- DEMO SCRIPT FOR NEW USERS ---
# Runs one detection simulation and plots the result. No input needed.

NRUNS = 1
STAR_CATALOG = 'Gaia'        # or 'ExoCat_1'
SIM_NAME = 'flat_universe'   # 'flat_universe', 'kepler', 'tess', 'rv', 'hwo', 'lifesim'

# flat_universe draws its own planets and takes under a minute. The others build a
# P-Pop universe per run, which takes about 20 minutes each.
sim_funcs = {
    'hwo': main_hwo,
    'lifesim': main_lifesim,
    'kepler': main_kepler,
    'tess': main_tess,
    'rv': main_rv,
}

def main():
    if SIM_NAME == 'flat_universe':
        print("\nRunning flat_universe: rocky planets around G, K and M stars through the "
              "TESS transit and HARPS/NIRPS RV detection models.\n")
        # Imported here because the module sets figure-wide font sizes on import.
        from plotting.scripts.analysis.multi import flat_transit_rv_3x3
        flat_transit_rv_3x3.main(paper_copy=False)
    elif SIM_NAME in sim_funcs:
        print(f"\nRunning {SIM_NAME} with {NRUNS} run(s), catalog {STAR_CATALOG!r}.\n")
        df = sim_funcs[SIM_NAME](parallel=True, nruns=np.arange(NRUNS),
                                 star_catalog=STAR_CATALOG, run_anew=True)
        print("Simulation done. Plotting.\n")
        plot_all(df=df, sim_name=SIM_NAME, nruns=NRUNS,
                 star_catalog=STAR_CATALOG, use_multiprocessing=True)
    else:
        raise SystemExit(f"Unknown simulation {SIM_NAME!r}. "
                         f"Options: {['flat_universe', *sim_funcs]}")

    print("Done. Plots are under results/figures/.")


# The guard is required: parallel runs start worker processes, and on macOS and
# Windows each worker re-imports this file. Without it every worker would start
# the simulation again and the run hangs.
if __name__ == '__main__':
    main()

"""Quickstart for the end-to-end exoplanet detection framework.

The default run visualizes a joint transit/RV selection function. Set ``SIM_NAME`` to run
one of the full transit, RV, coronagraph, or interferometer pipelines instead.
"""

from importlib import import_module

import numpy as np

# --- DEMO SCRIPT FOR NEW USERS ---
# Runs one detection simulation and plots the result. No input needed.

NRUNS = 1
STAR_CATALOG = 'Gaia'        # or 'ExoCat_1'
SIM_NAME = 'flat_universe'   # 'flat_universe', 'kepler', 'tess', 'rv', 'hwo', 'lifesim'

# flat_universe draws its own planets and takes under a minute. The others build a
# P-Pop universe per run, which takes about 20 minutes each.
SIMULATORS = {
    'kepler': 'run.kepler.run_kepler',
    'tess': 'run.tess.run_tess',
    'rv': 'run.rv.run_rv',
    'hwo': 'run.hwo.hwo_run_multiple',
    'lifesim': 'run.lifesim.lifesim_run_multiple',
}


def load_simulator(name):
    """Import only the selected detection pipeline."""
    module_name = SIMULATORS[name]
    return import_module(module_name).main


def main():
    if SIM_NAME == 'flat_universe':
        print("\nRunning flat_universe: rocky planets around G, K and M stars through the "
              "TESS transit and HARPS/NIRPS RV detection models.\n")
        # Keep the paper-figure machinery out of startup for the other pipelines.
        from plotting.scripts.analysis import flat_transit_rv_3x3
        flat_transit_rv_3x3.main(paper_copy=False)
    elif SIM_NAME in SIMULATORS:
        print(f"\nRunning {SIM_NAME} with {NRUNS} run(s), catalog {STAR_CATALOG!r}.\n")
        simulator = load_simulator(SIM_NAME)
        df = simulator(parallel=True, nruns=np.arange(NRUNS),
                       star_catalog=STAR_CATALOG, run_anew=True)
        print("Simulation done. Plotting.\n")
        from plotting.plot import plot_all
        plot_all(df=df, sim_name=SIM_NAME, nruns=NRUNS,
                 star_catalog=STAR_CATALOG, use_multiprocessing=True)
    else:
        options = ['flat_universe', *SIMULATORS]
        raise SystemExit(f"Unknown simulation {SIM_NAME!r}. "
                         f"Options: {options}")

    print("Done. Plots are under results/figures/.")


# The guard is required: parallel runs start worker processes, and on macOS and
# Windows each worker re-imports this file. Without it every worker would start
# the simulation again and the run hangs.
if __name__ == '__main__':
    main()

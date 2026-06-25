"""
run_sim_from1.py — top-up universe generator.

You already have kepler_catalog_0.csv and tess_catalog_0.csv (seed 0). This
script generates only the REMAINING universes (seeds 1 .. N_UNIVERSES-1), so you
end up with catalog_0..N-1 for both missions WITHOUT re-running seed 0.

Why this works: run_single(i) in run_Kepler.py / run_TESS.py uses `i` as BOTH
the RNG seed (np.random.default_rng(i)) and the output filename index
(catalog_{i}.csv). So NRUNS = np.arange(START, N_UNIVERSES) writes exactly
catalog_START..N-1 with seeds START..N-1, and never touches catalog_0.

plot=False on purpose: the DataFrame returned here only covers the seeds
generated in THIS run (1..N-1), not seed 0. The real stacked science figures
come from script 44, which reads catalog_0..N-1 together. Plotting a partial set
here would be misleading, so it is skipped.

Run from repo root:
    python run/run_sim_from1.py
"""

import os
import sys

# Make the repo root importable before pulling in the run.* packages (needed
# when this file is launched directly as `python run/run_sim_from1.py`).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from run.run_sim import run_sim, main_kepler, main_tess


if __name__ == "__main__":
    START = 1            # catalog_0 already exists for both missions -> start at 1
    N_UNIVERSES = 10     # final goal: catalog_0..9 (10 universes total)

    NRUNS = np.arange(START, N_UNIVERSES)   # seeds / indices 1..9

    print(f"Top-up run: generating universes {NRUNS.tolist()} "
          f"(catalog_0 kept as-is). {len(NRUNS)} universe(s) per mission.")

    run_sim(func=main_kepler, name="kepler", parallel=True, nruns=NRUNS,
            star_catalog="Gaia", run_anew=True, plot=False)   # Gaia 60pc
    print("Completed Kepler top-up")

    run_sim(func=main_tess, name="TESS", parallel=True, nruns=NRUNS,
            star_catalog="Gaia", run_anew=True, plot=False)   # Gaia 60pc
    print("Completed TESS top-up")

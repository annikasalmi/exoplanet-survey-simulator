"""
finish_tess_catalog_8_9.py — complete the TESS Gaia-60pc universe set.

tess_catalog_0..7 already exist in run/tess/data/Gaia_cdpp_v1/ (the directory
produced by tess_experiment="cdpp_v1" with star_catalog="Gaia"). This script
generates the two remaining universes (run indices 8 and 9) with the current
PPop generator + TESSData detector, using the exact same settings as the main
run_sim.py TESS call, and writes:

    run/tess/data/Gaia_cdpp_v1/tess_catalog_8.csv
    run/tess/data/Gaia_cdpp_v1/tess_catalog_9.csv

Each run_single(i) seeds PPop with np.random.default_rng(i), so indices 8 and 9
are independent Monte-Carlo realizations consistent with 0..7.

Run from the project root:
    python run/tess/finish_tess_catalog_8_9.py
"""

import os
import sys
import time

import numpy as np

# Add the project root to the Python path (same idiom as run/run_sim.py).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from run.tess.run_TESS import main as main_tess


# Only the two missing universes. Keep these as a list so it's trivial to
# re-run a single index (e.g. REMAINING = [9]) if one fails midway.
REMAINING = np.array([8, 9])


def main():
    start = time.time()
    print(f"Finishing TESS catalogs for runs {REMAINING.tolist()} (Gaia_cdpp_v1)...")

    # Mirror the run_sim.py TESS call: parallel, Gaia 60pc, run anew.
    # run_TESS.main() caps the pool at 2 workers internally (TESS workers are
    # RAM-heavy), so the two runs go side by side.
    df = main_tess(
        parallel=True,
        nruns=REMAINING,
        star_catalog="Gaia",
        run_anew=True,
        use_combined_sample=False,
        tess_experiment="cdpp_v1",
    )

    elapsed = time.time() - start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\nDone. Generated runs {REMAINING.tolist()} in {h}:{m:02d}:{s:02d}.")
    print(f"Total rows across new runs: {len(df)}")
    if "detected" in df.columns:
        print("detected value counts:")
        print(df["detected"].value_counts(dropna=False))
    if "tess_reason_category" in df.columns:
        print("tess_reason_category value counts:")
        print(df["tess_reason_category"].value_counts(dropna=False))


if __name__ == "__main__":
    main()

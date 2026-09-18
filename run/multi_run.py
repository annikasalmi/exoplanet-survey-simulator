"""The loop every P-Pop pipeline shares: run or reload seeded universes, then stack them."""

import multiprocessing as mp
from numbers import Integral
import time
from functools import partial

import pandas as pd


def normalize_nruns(nruns):
    """Return an explicit list of run ids from an int or iterable."""
    if isinstance(nruns, Integral):
        if nruns < 0:
            raise ValueError("nruns must be non-negative")
        return list(range(int(nruns)))
    return list(nruns)


def run_universes(run_single, load_single, nruns, star_catalog, run_anew, parallel, max_workers):
    """run_single(i) (or load_single(i) if not run_anew) per universe i in nruns; returns them
    stacked with a 'run' column."""
    nruns = normalize_nruns(nruns)
    start = time.time()
    runner = partial(run_single if run_anew else load_single, star_catalog=star_catalog)

    if parallel:
        # spawn on every OS, so workers start clean as on macOS and Windows.
        with mp.get_context('spawn').Pool(processes=min(len(nruns), max_workers)) as pool:
            results = pool.map(runner, nruns)
    else:
        results = [runner(i) for i in nruns]

    df_concat = pd.concat(results, keys=nruns).reset_index(level=0).rename(columns={'level_0': 'run'}).reset_index(drop=True)
    print(f"Total time: {time.time() - start:.2f} seconds")
    return df_concat

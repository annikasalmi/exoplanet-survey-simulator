"""The loop every P-Pop pipeline shares: run or reload seeded universes, then stack them."""

import multiprocessing as mp
from numbers import Integral
import time
from functools import partial
from pathlib import Path

import pandas as pd


def normalize_nruns(nruns):
    """Return an explicit list of run ids from an int or iterable."""
    if isinstance(nruns, Integral):
        if nruns < 0:
            raise ValueError("nruns must be non-negative")
        return list(range(int(nruns)))
    return list(nruns)


def run_universes(run_single, *, nruns=1, star_catalog='Gaia', run_anew=True,
                  parallel=False, max_workers=1, cache_dir=None, cache_prefix=None,
                  load_single=None):
    """Generate or reload seeded catalogs and stack them with a run column.

    CSV caches use cache_dir/star_catalog/{cache_prefix}_catalog_{i}.csv.
    Supply load_single only for a different format, such as LIFEsim's HDF catalog.
    """
    nruns = normalize_nruns(nruns)
    start = time.time()
    if not run_anew and load_single is None:
        results = [pd.read_csv(Path(cache_dir) / star_catalog / f'{cache_prefix}_catalog_{i}.csv')
                   for i in nruns]
    else:
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

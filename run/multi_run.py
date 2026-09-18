"""The loop every P-Pop pipeline shares: run or reload seeded universes, then stack them."""

import multiprocessing as mp
from numbers import Integral
from pathlib import Path
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


def run_universes(
    run_single,
    nruns,
    star_catalog,
    run_anew,
    parallel,
    max_workers,
    *,
    catalog_dir=None,
    catalog_stem=None,
    load_single=None,
):
    """Run or reload seeded universes, then return one stacked catalog."""
    nruns = normalize_nruns(nruns)
    start = time.time()
    if run_anew and parallel:
        runner = partial(run_single, star_catalog=star_catalog)
        # spawn on every OS, so workers start clean as on macOS and Windows.
        with mp.get_context('spawn').Pool(processes=min(len(nruns), max_workers)) as pool:
            results = pool.map(runner, nruns)
    elif run_anew:
        results = [run_single(i, star_catalog=star_catalog) for i in nruns]
    elif load_single is not None:
        results = [load_single(i, star_catalog=star_catalog) for i in nruns]
    elif catalog_dir is not None and catalog_stem is not None:
        directory = Path(catalog_dir) / star_catalog
        results = [pd.read_csv(directory / f"{catalog_stem}_{i}.csv") for i in nruns]
    else:
        raise ValueError("reloading requires load_single or catalog_dir and catalog_stem")

    df_concat = pd.concat(results, keys=nruns).reset_index(level=0).rename(columns={'level_0': 'run'}).reset_index(drop=True)
    print(f"Total time: {time.time() - start:.2f} seconds")
    return df_concat

from __future__ import annotations

import os
import sys
import time
import multiprocessing as mp
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import numpy as np
import pandas as pd

from run.ppop.ppop_generator import PPop

from lifesim.core.kepler_data import KeplerData
from tools.paths import KEPLER_DATA_DIR

from PPop.StarCatalogs import (
    CrossfieldBrightSample,
    ExoCat_1,
    LTC_2,
    LTC_3,
    gaia,
)



def set_star_catalog(PPopObj, star_catalog: str):
    """
    Copy the HWO star-catalog logic.
    This lets Kepler use the same input population as HWO.
    """

    if star_catalog == "CrossfieldBrightSample":
        PPopObj.StarCatalog = CrossfieldBrightSample
    elif star_catalog == "ExoCat_1":
        PPopObj.StarCatalog = ExoCat_1
    elif star_catalog == "LTC_3":
        PPopObj.StarCatalog = LTC_3
    elif star_catalog == "LTC_2":
        PPopObj.StarCatalog = LTC_2
    elif star_catalog == "Gaia":
        PPopObj.StarCatalog = gaia
    else:
        raise ValueError(f"Unknown star catalog: {star_catalog}")

    return PPopObj


def run_single(i, star_catalog="Gaia", use_combined_sample=False):
    """
    One Kepler run.

    This is the Kepler version of HWO's run_single().
    """

    print(f"Running Kepler for run {i} with star catalog {star_catalog}")

    rng = np.random.default_rng(i)

    PPopObj = PPop(rng=rng, use_combined_sample=use_combined_sample)
    PPopObj = set_star_catalog(PPopObj, star_catalog)

    filename = f"test_runs_kepler_{i}"
    data_path = os.path.join(KEPLER_DATA_DIR, filename)

    save_catalog = star_catalog
    save_dir = os.path.join(KEPLER_DATA_DIR, save_catalog)
    os.makedirs(save_dir, exist_ok=True)

    # 1. Make P-Pop planets
    df = PPopObj.run_ppop(data_path=data_path)

    # 2. Convert P-Pop output to the standard catalog columns
    PPopObj.catalog_from_ppop(data_path, df=df)

    # 3. Run Kepler detection math
    print(f"  running Kepler detection on {len(PPopObj.catalog)} planets...")
    kepler_data = KeplerData(PPopObj.catalog)
    kepler_data.determine_detectable()

    df = kepler_data.catalog

    # 4. Save result
    out_path = os.path.join(save_dir, f"kepler_catalog_{i}.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved Kepler catalog to: {out_path}")

    return df


def run_kepler_import_catalog(i, star_catalog, use_combined_sample=False):
    """
    Load an existing Kepler CSV instead of rerunning P-Pop.
    """

    path = os.path.join(
        KEPLER_DATA_DIR,
        star_catalog,
        f"kepler_catalog_{i}.csv",
    )

    df = pd.read_csv(path)

    return df


def main(parallel=False, nruns=np.arange(1), star_catalog="Gaia", run_anew=True, use_combined_sample=False):
    """
    Multiple Kepler runs.

    This mirrors HWO's main().
    """

    start = time.time()

    if run_anew:
        runner = partial(run_single, star_catalog=star_catalog, use_combined_sample=use_combined_sample)

        if parallel:
            # Cap workers at 5: this box has 16 GB RAM and each universe peaks
            # ~2-2.5 GB, so 5 workers (~10-12 GB) leaves headroom. mp.cpu_count()
            # here is 12 (6 cores x2 SMT) which would over-subscribe and swap.
            with mp.Pool(processes=min(5, mp.cpu_count())) as pool:
                results = pool.map(runner, nruns)
        else:
            results = [
                run_single(i=i, star_catalog=star_catalog, use_combined_sample=use_combined_sample)
                for i in nruns
            ]

    else:
        runner = partial(run_kepler_import_catalog, star_catalog=star_catalog, use_combined_sample=use_combined_sample)

        if parallel:
            with mp.Pool(processes=min(4, mp.cpu_count())) as pool:  # see note above: cap at 5 for 16 GB RAM
                results = pool.map(runner, nruns)
        else:
            results = [
                run_kepler_import_catalog(i=i, star_catalog=star_catalog, use_combined_sample=use_combined_sample)
                for i in nruns
            ]

    df_concat = (
        pd.concat(results, keys=nruns)
        .reset_index(level=0)
        .rename(columns={"level_0": "run"})
        .reset_index(drop=True)
    )

    print(f"Total time: {time.time() - start:.2f} seconds")

    return df_concat


if __name__ == "__main__":
    NRUNS = np.arange(1)
    STAR_CATALOG = "Gaia"

    df = main(
        parallel=False,
        nruns=NRUNS,
        star_catalog=STAR_CATALOG,
        run_anew=True,
    )

    print("Kepler run finished.")
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    if "detected_best" in df.columns:
        print("Detected best:")
        print(df["detected_best"].value_counts(dropna=False))

    if "detected_worst" in df.columns:
        print("Detected worst:")
        print(df["detected_worst"].value_counts(dropna=False))
import os
import time
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd

from PPop.StarCatalogs import CrossfieldBrightSample, ExoCat_1, LTC_2, LTC_3, gaia
from telescopes.tess.detection_model import TESSData
from run.ppop.ppop_generator import PPop
from tools.paths import TESS_DATA_DIR

# A TESS worker holds CDPP tables, tess-point and an exozodi KDE on top of the
# universe, 2.5-4 GB each; four or more ran out of memory on 16 GB.
MAX_WORKERS = 2

# Every star gets TESSData's default coverage (5 consecutive sectors) and noise
# from the binned SPOC CDPP table in telescopes/tess/data. P-Pop stars have no TIC IDs, so the per-TIC CDPP CSVs would go unused.
TESS_DEFAULTS = {
    "use_cdpp_tables": False,
    "min_transits": 2,
    "snr_threshold": 7.1,
    "phase_mode": "random",
    "tmag_limit": 16.0,
}

def set_star_catalog(ppop_obj, star_catalog: str):
    if star_catalog == "CrossfieldBrightSample":
        ppop_obj.StarCatalog = CrossfieldBrightSample
    elif star_catalog == "ExoCat_1":
        ppop_obj.StarCatalog = ExoCat_1
    elif star_catalog == "LTC_3":
        ppop_obj.StarCatalog = LTC_3
    elif star_catalog == "LTC_2":
        ppop_obj.StarCatalog = LTC_2
    elif star_catalog == "Gaia":
        ppop_obj.StarCatalog = gaia
    else:
        raise ValueError(f"Unknown star catalog: {star_catalog}")
    return ppop_obj


def run_single(i, star_catalog='Gaia'):
    '''
    Runs a single instance of the PPop simulation and TESS data analysis.
    '''
    print(f"Running TESS for run {i} with star catalog {star_catalog}")
    rng = np.random.default_rng(i)
    PPopObj = set_star_catalog(PPop(rng=rng), star_catalog)

    data_path = os.path.join(TESS_DATA_DIR, f'test_runs_tess_{i}')
    df = PPopObj.run_ppop(data_path=data_path)
    PPopObj.catalog_from_ppop(data_path, df=df)
    PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.0)

    # random_seed fixes each run's transit phases, so reruns reproduce.
    tess_data = TESSData(PPopObj.catalog, source="ppop", random_seed=i, **TESS_DEFAULTS)
    tess_data.determine_detectable()

    df = tess_data.catalog
    save_dir = os.path.join(TESS_DATA_DIR, star_catalog)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f'tess_catalog_{i}.csv'), index=False)

    return df

def run_tess_import_catalog(i, star_catalog):
    df = pd.read_csv(os.path.join(TESS_DATA_DIR, star_catalog, f'tess_catalog_{i}.csv'))
    return df

def main(parallel=False, nruns=np.arange(1), star_catalog='Gaia', run_anew=True):
    start = time.time()

    if run_anew:
        runner = partial(run_single, star_catalog=star_catalog)
    else:
        runner = partial(run_tess_import_catalog, star_catalog=star_catalog)

    if parallel:
        with mp.Pool(processes=min(len(nruns), MAX_WORKERS)) as pool:
            results = pool.map(runner, nruns)
    else:
        results = [runner(i) for i in nruns]

    df_concat = pd.concat(results, keys=nruns).reset_index(level=0).rename(columns={'level_0': 'run'}).reset_index(drop=True)

    print(f"Total time: {time.time() - start:.2f} seconds")

    return df_concat


if __name__ == '__main__':
    main()

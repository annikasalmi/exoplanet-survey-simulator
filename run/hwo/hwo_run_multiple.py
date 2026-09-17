import time
import os
import pandas as pd
import multiprocessing as mp
import numpy as np
from functools import partial

from telescopes.hwo.detection_model import HWOData
from run.ppop.ppop_generator import PPop, set_star_catalog
from tools.paths import HWO_DATA_DIR

def run_single(i, star_catalog='Gaia'):
    '''
    Runs a single instance of the PPop simulation and HWO data analysis.
    '''
    print(f"Running HWO for run {i} with star catalog {star_catalog}")
    rng = np.random.default_rng(i)
    PPopObj = set_star_catalog(PPop(rng=rng), star_catalog)

    filename = f'test_runs_hwo_{i}'
    data_path = os.path.join(HWO_DATA_DIR, filename)

    df = PPopObj.run_ppop(data_path=data_path)
    PPopObj.catalog_from_ppop(data_path, df=df)
    PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.0)
    # PPopObj.catalog_remove_distance(stype='M', mode='larger', dist=10.0)

    hwo_data = HWOData(PPopObj.catalog)
    hwo_data.determine_detectable()

    df = hwo_data.catalog
    save_dir = os.path.join(HWO_DATA_DIR, star_catalog)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f'hwo_catalog_{i}.csv'), index=False)

    return df

def run_hwo_import_catalog(i, star_catalog):
    df = pd.read_csv(os.path.join(HWO_DATA_DIR, star_catalog, f'hwo_catalog_{i}.csv'))
    return df

def main(parallel=False, nruns=np.arange(1), star_catalog='Gaia', run_anew=True):
    start = time.time()

    if run_anew:
        runner = partial(run_single, star_catalog=star_catalog)
        if parallel:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.map(runner, nruns)
        else:
            results = [run_single(i=i, star_catalog=star_catalog) for i in nruns]
    else:
        runner = partial(run_hwo_import_catalog, star_catalog=star_catalog)
        if parallel:
            with mp.Pool(processes=mp.cpu_count()) as pool:
                results = pool.map(runner, nruns)
        else:
            results = [run_hwo_import_catalog(i=i, star_catalog=star_catalog) for i in nruns]

    df_concat = pd.concat(results, keys=nruns).reset_index(level=0).rename(columns={'level_0': 'run'}).reset_index(drop=True)
    print(f"Total time: {time.time() - start:.2f} seconds")

    return df_concat


if __name__ == '__main__':
    NRUNS = np.arange(3)
    STAR_CATALOG = 'Gaia'#ExoCat_1'  # or 'LTC_3'
    mp.set_start_method('spawn')
    main(nruns=NRUNS, star_catalog=STAR_CATALOG, parallel=True)
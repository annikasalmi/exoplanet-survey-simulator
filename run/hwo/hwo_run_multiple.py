import time
import os
import pandas as pd
import multiprocessing as mp
import numpy as np
from functools import partial

from PPop.StarCatalogs import CrossfieldBrightSample, ExoCat_1, LTC_2, LTC_3, gaia
from lifesim.core.hwo_data import HWOData
from run.ppop.ppop_generator import PPop
from tools.paths import HWO_DATA_DIR
from plot.plot import plot_by_star, plot_by_planet

def run_single(i, star_catalog='Gaia'):
    '''
    Runs a single instance of the PPop simulation and HWO data analysis.
    '''
    rng = np.random.default_rng()
    PPopObj = PPop(rng=rng)

    if star_catalog == 'CrossfieldBrightSample':
        PPopObj.StarCatalog = CrossfieldBrightSample
    elif star_catalog == 'ExoCat_1':
        PPopObj.StarCatalog = ExoCat_1
    elif star_catalog == 'LTC_3':
        PPopObj.StarCatalog = LTC_3
    elif star_catalog == 'LTC_2':
        PPopObj.StarCatalog = LTC_2
    elif star_catalog == 'Gaia':
        PPopObj.StarCatalog = gaia

    filename = f'test_runs_hwo_{i}'
    data_path = os.path.join(HWO_DATA_DIR, filename)

    df = PPopObj.run_ppop(data_path=data_path)
    PPopObj.catalog_from_ppop(data_path, df=df)
    PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.0)
    PPopObj.catalog_remove_distance(stype='M', mode='larger', dist=10.0)

    hwo_data = HWOData(PPopObj.catalog)
    hwo_data.determine_detectable()

    df = hwo_data.catalog

    return df

def main(parallel=False, nruns=1, star_catalog='Gaia'):
    start = time.time()

    runner = partial(run_single, star_catalog=star_catalog)
    if parallel:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(runner, range(nruns))
    else:
        results = [run_single(i=i, star_catalog=star_catalog) for i in range(nruns)]

    df_concat = pd.concat(results, keys=range(nruns)).reset_index(level=0).rename(columns={'level_0': 'run'})

    plot_by_star(df_concat, nruns=nruns, star_catalog=star_catalog)
    plot_by_planet(df_concat, nruns=nruns, star_catalog=star_catalog)
    print(f"Total time: {time.time() - start:.2f} seconds")


if __name__ == '__main__':
    NRUNS = 3
    STAR_CATALOG = 'Gaia'#ExoCat_1'  # or 'LTC_3'
    mp.set_start_method('spawn')
    main(nruns=NRUNS, star_catalog=STAR_CATALOG, parallel=True)
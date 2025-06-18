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
    
    bins = [0, 1.5, 3.0, 6.0]
    labels = ['<1.5', '1.5–3.0', '3.0–6.0']
    df['radius_bin'] = pd.cut(df['radius_p'], bins=bins, labels=labels, include_lowest=True)

    # Add "Rocky HZ" bin
    rocky_hz = df[(df['habitable'] == True) & (df['radius_p'] < 1.5)].copy()
    rocky_hz['radius_bin'] = 'Rocky HZ'

    # Combine all rows
    df_all = pd.concat([df, rocky_hz], ignore_index=True)

    # Group by star type and radius bin
    grouped_df = df_all.groupby(['stype', 'radius_bin']).size().reset_index(name='count')

    return grouped_df

def main(parallel=False, nruns=1, star_catalog='Gaia'):
    start = time.time()

    runner = partial(run_single, star_catalog=star_catalog)
    if parallel:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(runner, range(nruns))
    else:
        results = [run_single(i=i, star_catalog=star_catalog) for i in range(nruns)]

    df_all = pd.concat(results, keys=range(nruns)).reset_index(level=0).rename(columns={'level_0': 'run'})

    plot_by_star(df_all, nruns=nruns, star_catalog=star_catalog)
    plot_by_planet(df_all, nruns=nruns, star_catalog=star_catalog)
    print(f"Total time: {time.time() - start:.2f} seconds")


if __name__ == '__main__':
    NRUNS = 3
    STAR_CATALOG = 'Gaia'#ExoCat_1'  # or 'LTC_3'
    mp.set_start_method('spawn')
    main(nruns=NRUNS, star_catalog=STAR_CATALOG, parallel=True)
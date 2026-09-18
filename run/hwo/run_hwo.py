import os
import pandas as pd
import numpy as np

from science.populations.ppop import PPop, set_star_catalog
from science.telescopes.hwo.detection_model import HWOData
from run.multi_run import run_universes
from tools.paths import HWO_DATA_DIR

# Universes run at once; defaults to every core.
MAX_WORKERS = os.cpu_count()

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
    return run_universes(run_single, run_hwo_import_catalog, nruns, star_catalog,
                         run_anew, parallel, MAX_WORKERS)


if __name__ == '__main__':
    NRUNS = np.arange(3)
    STAR_CATALOG = 'Gaia'#ExoCat_1'  # or 'LTC_3'
    main(nruns=NRUNS, star_catalog=STAR_CATALOG, parallel=True)

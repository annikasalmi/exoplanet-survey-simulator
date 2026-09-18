import os

import numpy as np

from science.populations.universes.ppop import PPop
from science.telescopes.tess.detection_model import TESSData
from run.multi_run import run_universes
from tools.paths import TESS_DATA_DIR

MAX_WORKERS = 2

TESS_DEFAULTS = {
    "use_cdpp_tables": False,
    "min_transits": 2,
    "snr_threshold": 7.1,
    "phase_mode": "random",
    "tmag_limit": 16.0,
}


def run_single(i, star_catalog='Gaia'):
    print(f"Running TESS for run {i} with star catalog {star_catalog}")
    rng = np.random.default_rng(i)
    population = PPop(rng=rng, star_catalog=star_catalog)

    data_path = os.path.join(TESS_DATA_DIR, f'test_runs_tess_{i}')
    df = population.run_ppop(data_path=data_path)
    population.catalog_from_ppop(data_path, df=df)
    population.catalog_remove_distance(stype='A', mode='larger', dist=0.0)

    tess_data = TESSData(population.catalog, source="ppop", random_seed=i, **TESS_DEFAULTS)
    df = tess_data.determine_detectable()

    save_dir = os.path.join(TESS_DATA_DIR, star_catalog)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f'tess_catalog_{i}.csv'), index=False)
    return df


if __name__ == '__main__':
    run_universes(run_single, cache_dir=TESS_DATA_DIR,
                  cache_prefix='tess', max_workers=MAX_WORKERS)

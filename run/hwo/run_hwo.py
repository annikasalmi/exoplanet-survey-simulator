import os
import numpy as np

from science.populations.universes.ppop import PPop
from science.telescopes.hwo.detection_model import HWOData
from run.multi_run import run_universes
from tools.paths import HWO_DATA_DIR

MAX_WORKERS = os.cpu_count()


def run_single(i, star_catalog='Gaia'):
    print(f"Running HWO for run {i} with star catalog {star_catalog}")
    rng = np.random.default_rng(i)
    population = PPop(rng=rng, star_catalog=star_catalog)

    data_path = os.path.join(HWO_DATA_DIR, f'test_runs_hwo_{i}')
    df = population.run_ppop(data_path=data_path)
    population.catalog_from_ppop(data_path, df=df)
    population.catalog_remove_distance(stype='A', mode='larger', dist=0.0)

    df = HWOData(population.catalog).determine_detectable()
    save_dir = os.path.join(HWO_DATA_DIR, star_catalog)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f'hwo_catalog_{i}.csv'), index=False)
    return df


if __name__ == '__main__':
    run_universes(run_single, nruns=np.arange(3), parallel=True, cache_dir=HWO_DATA_DIR,
                  cache_prefix='hwo', max_workers=MAX_WORKERS)

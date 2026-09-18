import os
from functools import partial

import numpy as np
from science.populations.ppop import PPop, set_star_catalog
from science.telescopes.detection import run_rv_best
from run.multi_run import run_universes
from tools.paths import RV_DATA_DIR

# Each universe peaks at 2-2.5 GB, as for Kepler; more workers than this swaps on 16 GB.
MAX_WORKERS = 5


def run_single(i, star_catalog='Gaia'):
    '''
    Runs a single instance of the PPop simulation and RV data analysis.
    '''
    print(f"Running RV for run {i} with star catalog {star_catalog}")
    rng = np.random.default_rng(i)
    PPopObj = set_star_catalog(PPop(rng=rng), star_catalog)

    data_path = os.path.join(RV_DATA_DIR, f'test_runs_rv_{i}')
    df = PPopObj.run_ppop(data_path=data_path)
    PPopObj.catalog_from_ppop(data_path, df=df)
    PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.0)

    # A planet counts as detected if HARPS (optical) or NIRPS (near-IR, which sees
    # M dwarfs better) reaches it: the per-planet best spectrograph.
    df = run_rv_best(PPopObj.catalog)

    save_dir = os.path.join(RV_DATA_DIR, star_catalog)
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(os.path.join(save_dir, f'rv_catalog_{i}.csv'), index=False)

    return df

main = partial(
    run_universes,
    run_single,
    nruns=np.arange(1),
    star_catalog="Gaia",
    run_anew=True,
    parallel=False,
    max_workers=MAX_WORKERS,
    catalog_dir=RV_DATA_DIR,
    catalog_stem="rv_catalog",
)
main.__name__ = "main"


if __name__ == '__main__':
    main()

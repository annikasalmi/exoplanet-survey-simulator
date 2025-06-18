import os
import lifesim
import numpy as np
from run.ppop.ppop_generator import PPop
import multiprocessing as mp
import time
import pandas as pd
from functools import partial

from tools.paths import PPOP_DATA_DIR, LIFESIM_DATA_DIR
from PPop.StarCatalogs import CrossfieldBrightSample, ExoCat_1, LTC_2, LTC_3, gaia

RUN_PPOP = False


def main(parallel=False, nruns=1, star_catalog='Gaia'):
    start=time.time()

    runner = partial(run_lifesim_single, star_catalog=star_catalog)
    if parallel:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(runner, range(nruns))
    else:
        results = [run_lifesim_single(i=i, star_catalog=star_catalog) for i in range(nruns)]
    print(f"Finished {nruns} runs in {time.time() - start:.2f} seconds")
    print('Starting plotting...')

    # Step 2: Combine all runs into one DataFrame
    df_concat = pd.concat(results, keys=range(nruns)).reset_index(level=0).rename(columns={'level_0': 'run'})

    print(f"Total time: {time.time() - start:.2f} seconds")

    return df_concat

if __name__ == '__main__':
    
    mp.set_start_method('spawn')
    main()


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

def run_lifesim_single(i, star_catalog='Gaia'):
    rng = np.random.default_rng()
    # ----- Generate new planet population -----
    PPopObj = PPop(rng=i)

    if star_catalog == 'ExoCat_1':
        PPopObj.StarCatalog = ExoCat_1
    elif star_catalog == 'LTC_3':
        PPopObj.StarCatalog = LTC_3
    elif star_catalog == 'Gaia':
        PPopObj.StarCatalog = gaia

    filename = f'test_runs_lifesim_{i}'
    data_path = os.path.join(PPOP_DATA_DIR, filename)

    df = PPopObj.run_ppop(data_path=data_path)
    PPopObj.catalog_from_ppop(data_path, df=df)
    PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.0)
    # PPopObj.catalog_remove_distance(stype='M', mode='larger', dist=10.0)

    # ----- Run LIFEsim with this catalog -----
    bus = lifesim.Bus()
    bus.data.options.set_scenario('baseline')
    bus.data.options.set_manual(diameter=4.0)
    bus.data.options.set_manual(output_path=LIFESIM_DATA_DIR)
    bus.data.options.set_manual(output_filename=f'/test_runs_{i}')
    bus.data.catalog_from_ppop(data_path, df=df)

    # ----- Instrument and Modules -----
    instrument = lifesim.Instrument(name='inst', rng=i)
    bus.add_module(instrument)
    bus.add_module(lifesim.TransmissionMap(name='transm'))
    bus.add_module(lifesim.PhotonNoiseExozodi(name='exo'))
    bus.add_module(lifesim.PhotonNoiseLocalzodi(name='local'))
    bus.add_module(lifesim.PhotonNoiseStar(name='star'))
    bus.connect(('inst', 'transm'))
    bus.connect(('inst', 'exo'))
    bus.connect(('inst', 'local'))
    bus.connect(('inst', 'star'))
    bus.connect(('star', 'transm'))

    opt = lifesim.Optimizer(name='opt')
    ahgs = lifesim.AhgsModule(name='ahgs')
    bus.add_module(opt)
    bus.add_module(ahgs)
    bus.connect(('transm', 'opt'))
    bus.connect(('inst', 'opt'))
    bus.connect(('opt', 'ahgs'))

    # ----- Run Simulation -----
    instrument.get_snr()
    opt.ahgs()
    bus.save()

    return bus.data.catalog

def main(parallel=True, nruns=1, star_catalog='Gaia'):
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
    df_concat = pd.concat(results, ignore_index=True)

    print(f"Total time: {time.time() - start:.2f} seconds")

if __name__ == '__main__':
    
    mp.set_start_method('spawn')
    main()


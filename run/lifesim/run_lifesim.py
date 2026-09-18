import os
import sys
import numpy as np

# Add the lifesim directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import lifesim
from science.populations.ppop import PPop, set_star_catalog
import pandas as pd
import numpy as np

from run.multi_run import run_universes
from tools.paths import PPOP_DATA_DIR, LIFESIM_DATA_DIR
from plotting.plot import plot_all

RUN_PPOP = False
# Universes run at once; defaults to every core.
MAX_WORKERS = os.cpu_count()

def run_lifesim_single(i, star_catalog='Gaia'):
    print(f"Running LIFEsim for run {i} with star catalog {star_catalog}")
    rng = np.random.default_rng(i)
    # ----- Generate new planet population -----
    PPopObj = set_star_catalog(PPop(rng=rng), star_catalog)

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
    out_dir = os.path.join(LIFESIM_DATA_DIR, star_catalog)
    os.makedirs(out_dir, exist_ok=True)
    bus.data.options.set_manual(output_path=out_dir)
    bus.data.options.set_manual(output_filename=f'/test_runs_{i}')
    bus.data.catalog_from_ppop(data_path, df=df)

    # ----- Instrument and Modules -----
    instrument = lifesim.Instrument(name='inst', rng=rng)
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


def run_lifesim_import_catalog(i, star_catalog='Gaia'):
    rng = np.random.default_rng(i)

    # ----- Run LIFEsim with this catalog -----
    bus = lifesim.Bus()
    bus.data.options.set_scenario('baseline')
    bus.data.options.set_manual(diameter=4.0)
    os.makedirs(LIFESIM_DATA_DIR, exist_ok=True)
    bus.data.options.set_manual(output_path=LIFESIM_DATA_DIR)
    bus.data.options.set_manual(output_filename=f'/test_runs_{i}')

    bus.data.import_catalog(input_path=os.path.join(LIFESIM_DATA_DIR, star_catalog, f'test_runs_{i}_catalog.hdf5'))

    return bus.data.catalog

def main(parallel=True, nruns=np.arange(1), star_catalog='Gaia', run_anew=True):
    return run_universes(run_lifesim_single, run_lifesim_import_catalog, nruns, star_catalog,
                         run_anew, parallel, MAX_WORKERS)


if __name__ == '__main__':
    main()

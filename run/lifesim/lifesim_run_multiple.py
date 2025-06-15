import os
import lifesim
from run.ppop.ppop_generator import PPop
import multiprocessing as mp

from tools.paths import PPOP_DATA_DIR, LIFESIM_DATA_DIR
from PPop.StarCatalogs import CrossfieldBrightSample, ExoCat_1, LTC_2, LTC_3

NRUNS = 5  # or 500
STAR_CATALOG = 'LTC_3'#ExoCat_1'  # or 'LTC_3'

def run_lifesim_single(i):

    # ----- Generate new planet population -----
    PPopObj = PPop(seed=i)

    if STAR_CATALOG == 'ExoCat_1':
        PPopObj.StarCatalog = ExoCat_1
    elif STAR_CATALOG == 'LTC_3':
        PPopObj.StarCatalog = LTC_3

    filename = f'test_runs_hwo_{i}'
    data_path = os.path.join(PPOP_DATA_DIR, filename)

    df = PPopObj.run_ppop(seed=i, data_path=data_path)
    PPopObj.catalog_from_ppop(data_path, df=df)
    PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.0)
    PPopObj.catalog_remove_distance(stype='M', mode='larger', dist=10.0)
    catalog_path = f"{data_path}.txt"
    with open(catalog_path, 'w') as f:
        f.write(df.to_csv(sep='\t', index=False))

    output = os.path.join(LIFESIM_DATA_DIR, f'test_runs_{i}')

    # ----- Run LIFEsim with this catalog -----
    bus = lifesim.Bus()
    bus.data.options.set_scenario('baseline')
    bus.data.options.set_manual(diameter=4.0)
    bus.data.options.set_manual(output_path=output)
    bus.data.options.set_manual(output_filename=output)
    bus.data.catalog_from_ppop(input_path=catalog_path)
    bus.data.catalog_remove_distance(stype='A', mode='larger', dist=0.0)
    bus.data.catalog_remove_distance(stype='M', mode='larger', dist=10.0)

    # ----- Instrument and Modules -----
    instrument = lifesim.Instrument(name='inst', seed=i)
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

    ## SOMEHOW SAVE THIS AND READ THE DATA>>>>>
    bus_read = lifesim.Bus()
    bus_read.build_from_config(output+'.yaml')
    bus_read.data.import_catalog(input_path=output+'_catalog.hdf5')

def main(parallel=False):
    if parallel:
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(run_lifesim_single, range(NRUNS))
    else:
        results = [run_lifesim_single(i) for i in range(NRUNS)]

    for r in results:
        print(r)

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()


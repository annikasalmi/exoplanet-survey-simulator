import os
import sys
import numpy as np
import pandas as pd
import time

# Add the lifesim directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import lifesim

from tools.paths import LIFESIM_DATA_DIR
from tools.exoplanet_catalog import load_and_filter_exoplanets

from run.run_sim import plot_all

def load_exoplanets_csv(csv_path='exoplanets_2026.csv'):
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), '..', '..', csv_path)
    df = load_and_filter_exoplanets(csv_path, instrument='LIFE')
    
    # Create star objects for LIFEsim
    star_objects = []
    for i in range(len(df)):
        star_obj = type('Star', (), {
            'Rad': df['st_rad'].iloc[i],
            'Teff': df['st_teff'].iloc[i],
            'Dist': df['sy_dist'].iloc[i],
            'RA': 0.0,
            'Dec': 0.0,
            'Stype': df['stype'].iloc[i],
            'Name': f"Star_{i}"
        })()
        star_objects.append(star_obj)
    df['Star'] = star_objects
    
    return df

def run_lifesim_exoplanets(i, df_exoplanets, star_catalog='Exoplanets'):
    rng = np.random.default_rng(i)
    bus = lifesim.Bus()
    bus.data.options.set_scenario('baseline')
    bus.data.options.set_manual(diameter=4.0)
    bus.data.options.set_manual(output_path=os.path.join(LIFESIM_DATA_DIR, star_catalog))
    bus.data.options.set_manual(output_filename=f'/exoplanets_runs_{i}')
    df_exoplanets = df_exoplanets.reset_index(drop=True)
    bus.data.catalog_from_ppop(data_path="dummy", df=df_exoplanets)
    bus.data.catalog_remove_distance(stype='A', mode='larger', dist=0.)
    bus.data.catalog_remove_distance(stype='M', mode='larger', dist=10.)
    instrument = lifesim.Instrument(name='inst', rng=rng)
    bus.add_module(instrument)
    transm = lifesim.TransmissionMap(name='transm')
    bus.add_module(transm)
    exo = lifesim.PhotonNoiseExozodi(name='exo')
    bus.add_module(exo)
    local = lifesim.PhotonNoiseLocalzodi(name='local')
    bus.add_module(local)
    star = lifesim.PhotonNoiseStar(name='star')
    bus.add_module(star)
    bus.connect(('inst', 'transm'))
    bus.connect(('inst', 'exo'))
    bus.connect(('inst', 'local'))
    bus.connect(('inst', 'star'))
    bus.connect(('star', 'transm'))
    opt = lifesim.Optimizer(name='opt')
    bus.add_module(opt)
    ahgs = lifesim.AhgsModule(name='ahgs')
    bus.add_module(ahgs)
    bus.connect(('transm', 'opt'))
    bus.connect(('inst', 'opt'))
    bus.connect(('opt', 'ahgs'))
    instrument.get_snr()
    opt.ahgs()
    return bus.data.catalog

def main(nruns=np.arange(1), csv_path='exoplanets_2026.csv', plot=False, sim_name='LIFEsim_exoplanets', star_catalog='Exoplanets'):
    start = time.time()
    df_exoplanets = load_exoplanets_csv(csv_path)
    results = [run_lifesim_exoplanets(i=i, df_exoplanets=df_exoplanets) for i in nruns]
    valid_results = [r for r in results if r is not None]
    if valid_results:
        dfs_with_run = []
        for i, result in enumerate(valid_results):
            if result is not None:
                result_df = result.copy()
                result_df['run'] = nruns[i]
                dfs_with_run.append(result_df)
        df_concat = pd.concat(dfs_with_run, ignore_index=True)
    else:
        df_concat = pd.DataFrame()
    if plot:
        plot_all(df=df_concat, sim_name=sim_name, nruns=len(nruns), star_catalog=star_catalog, use_multiprocessing=False)
    return df_concat

if __name__ == '__main__':
    main(plot=True)
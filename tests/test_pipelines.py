"""End-to-end checks for the batch pipelines, on a small star catalog so CI
finishes in minutes. Each P-Pop pipeline must run, plot, reload its saved catalogs, and
reproduce a seeded universe. All output goes to a temp dir, never to results/.
"""

import matplotlib
matplotlib.use('Agg')

import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import plotting.base_plotter as base_plotter
import plotting.likelihood_ratio_plotter as likelihood_ratio_plotter
from plotting.plot import plot_all
import run.flat_universe.run_flat_universe as flat
from science.populations.universes.flat_curves import is_super_earth
import run.hwo.run_hwo as hwo
import run.kepler.run_kepler as kepler
import run.lifesim.run_lifesim as lifesim
import run.rv.run_rv as rv
import run.tess.run_tess as tess
from science.populations.universes.ppop import PPop
from run.multi_run import normalize_nruns, run_universes
from tools import paths

# Gaia stars within 10 pc: ~220 stars and ~700 planets per universe, a few seconds
# each, and still enough planets that Kepler and TESS detect some.
SMALL_DIST_RANGE = [0., 10.]
NRUNS = np.arange(2)

# name -> (module, output-dir attributes to redirect, detection column)
PPOP_PIPELINES = {
    'kepler': (kepler, ['KEPLER_DATA_DIR'], 'detected_best'),
    'tess': (tess, ['TESS_DATA_DIR'], 'detected_best'),
    'rv': (rv, ['RV_DATA_DIR'], 'detected_best'),
    'hwo': (hwo, ['HWO_DATA_DIR'], 'detected_best'),
    'lifesim': (lifesim, ['PPOP_DATA_DIR', 'LIFESIM_DATA_DIR'], 'detected'),
}

PLANET_COLUMNS = {'radius_p', 'mass_p', 'p_orb', 'semimajor_p', 'temp_p',
                  'distance_s', 'stype', 'habitable'}

# pandas 3 turns chained writes (df.col.iat[i] = x) into silent no-ops; this once left
# every LIFEsim SNR at 0. Fail on the warning instead.
if hasattr(pd.errors, 'ChainedAssignmentError'):
    pytestmark = pytest.mark.filterwarnings('error::pandas.errors.ChainedAssignmentError')


def _csv_comparable(got, expected):
    """The columns of both frames that survive a CSV round trip, text compared as str.
    P-Pop catalogs also hold Star objects, bound methods and the RNG, which CSV stores
    as '<... at 0x...>', and CSV reads text Gaia source IDs back as ints."""
    def text(s):
        return s.map(lambda v: '' if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    got_cols, expected_cols = {}, {}
    for col in expected.columns:
        s = expected[col]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            got_cols[col], expected_cols[col] = got[col], s
        elif s.map(lambda v: v is None or isinstance(v, (str, float, int, np.number))).all():
            got_cols[col], expected_cols[col] = text(got[col]), text(s)
    return (pd.DataFrame(got_cols).reset_index(drop=True),
            pd.DataFrame(expected_cols).reset_index(drop=True))


class SmallPPop(PPop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Dist_range = list(SMALL_DIST_RANGE)


@pytest.fixture(scope='session')
def sandbox(tmp_path_factory):
    """Point every pipeline and plotter at a temp dir and at the small star catalog."""
    root = tmp_path_factory.mktemp('results')
    with pytest.MonkeyPatch.context() as mp:
        for name, (module, dir_attrs, _) in PPOP_PIPELINES.items():
            mp.setattr(module, 'PPop', SmallPPop)
            for attr in dir_attrs:
                mp.setattr(module, attr, str(root / name))
        mp.setattr(flat, 'FLAT_CACHE_DIR', root / 'flat_universe')
        (root / 'flat_universe').mkdir()
        mp.setattr(base_plotter, 'PLOTS_DIR', str(root / 'figures'))
        mp.setattr(likelihood_ratio_plotter, 'OUT_DIR', str(root / 'likelihood_ratio'))
        yield root


@pytest.fixture(scope='session', params=list(PPOP_PIPELINES))
def ppop_run(request, sandbox):
    """Generate and plot two universes per pipeline, shared by the tests below."""
    name = request.param
    module, dir_attrs, detected_col = PPOP_PIPELINES[name]
    settings = dict(run_single=module.run_single, star_catalog='Gaia',
                    cache_dir=getattr(module, dir_attrs[-1]), cache_prefix=name,
                    load_single=lifesim.run_lifesim_import_catalog if name == 'lifesim' else None)
    df = run_universes(**settings, nruns=NRUNS)
    plot_all(df, sim_name=name, nruns=len(NRUNS), star_catalog='Gaia', use_multiprocessing=False)
    return name, settings, detected_col, df


def test_ppop_pipeline_runs(ppop_run):
    name, _, detected_col, df = ppop_run
    assert sorted(df['run'].unique()) == list(NRUNS)
    missing = (PLANET_COLUMNS | {detected_col}) - set(df.columns)
    assert not missing, f'{name} output lacks {sorted(missing)}'
    assert df['radius_p'].gt(0).all() and df['mass_p'].gt(0).all()


def test_ppop_pipeline_detects(ppop_run):
    name, _, detected_col, df = ppop_run
    assert df[detected_col].astype(bool).any(), f'{name} detected no planets'


def test_ppop_pipeline_plots(ppop_run, sandbox):
    name = ppop_run[0]
    pngs = list((sandbox / 'figures').glob(f'{name}_{len(NRUNS)}_Gaia/**/*.png'))
    assert pngs, f'plot_all wrote no figures for {name}'


def test_ppop_pipeline_reloads_saved_catalogs(ppop_run):
    name, settings, _, df = ppop_run
    reloaded = run_universes(**settings, nruns=NRUNS, run_anew=False)
    got, expected = _csv_comparable(reloaded, df)
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_ppop_pipeline_is_seeded(ppop_run):
    name, settings, _, df = ppop_run
    again = run_universes(**settings, nruns=NRUNS[1:])
    got, expected = _csv_comparable(again, df[df['run'] == NRUNS[1]])
    pd.testing.assert_frame_equal(got, expected)


def test_flat_universe(sandbox):
    n_planets = 20_000
    df = flat.main(seed=0, n_planets=n_planets, run_anew=True)

    a, b = df[df['universe_type'] == 'A'], df[df['universe_type'] == 'B']
    assert 0 < len(a) < len(b) <= n_planets
    assert not is_super_earth(a['mass_p'], a['radius_p']).any(), 'universe A must drop the super-Earths'
    assert is_super_earth(b['mass_p'], b['radius_p']).any()
    for col in ('kepler_detected', 'tess_detected', 'rv_detected'):
        assert 0 < df[col].sum() < len(df), f'{col} is all-or-nothing'

    # run_anew=False reads the cache. Truncating the cache then shows run_anew=True ignores
    # it and redraws the same universe from the seed.
    cached = flat.main(seed=0, n_planets=n_planets, run_anew=False)
    for path in (sandbox / 'flat_universe').glob('*.csv'):
        pd.read_csv(path).head(10).to_csv(path, index=False)
    assert len(flat.main(seed=0, n_planets=n_planets, run_anew=False)) == 20
    regenerated = flat.main(seed=0, n_planets=n_planets, run_anew=True)
    pd.testing.assert_frame_equal(cached, df, check_dtype=False)
    pd.testing.assert_frame_equal(regenerated, df, check_dtype=False)

    likelihood_ratio_plotter.main(df)
    assert list((sandbox / 'likelihood_ratio').glob('*.png'))


def test_kepler_on_nasa_pscomppars(sandbox):
    out_csv = sandbox / 'kepler_nasa.csv'
    df = kepler.run_nasa_pscomppars(output_csv=out_csv)
    assert out_csv.exists()
    assert len(df) > 100
    assert 0 < df['detected'].sum() < len(df)


def test_integer_nruns():
    assert normalize_nruns(3) == [0, 1, 2]


def test_ppop_quickstart(tmp_path, monkeypatch):
    import science.populations.universes.ppop as ppop
    import plotting.plot_population as plotter

    monkeypatch.setattr(ppop, 'PPop', SmallPPop)
    monkeypatch.setattr(paths, 'CATALOGS_DIR', str(tmp_path / 'catalogs'))
    monkeypatch.setattr(plotter, 'PLOTS_DIR', str(tmp_path / 'plots'))
    monkeypatch.setattr(sys, 'argv', ['sim.py', '--universe', 'ppop', '--star-catalog', 'Gaia'])
    runpy.run_path(str(Path(paths.REPO_ROOT) / 'sim.py'), run_name='__main__')

    df = pd.read_csv(tmp_path / 'catalogs' / 'ppop' / 'ppop_Gaia_seed0.csv')
    assert len(df) > 0
    assert df[['kepler_detected', 'tess_detected', 'rv_detected']].any().all()
    assert {p.name for p in (tmp_path / 'plots').rglob('*.png')} == {'population.png', 'detections.png'}

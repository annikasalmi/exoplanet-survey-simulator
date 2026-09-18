"""End-to-end checks for every pipeline run_sim can drive, on a small star catalog so CI
finishes in minutes. Each P-Pop pipeline must run, plot, reload its saved catalogs, and
reproduce a seeded universe. All output goes to a temp dir, never to results/.
"""

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

import plotting.base_plotter as base_plotter
import plotting.likelihood_ratio_plotter as likelihood_ratio_plotter
from plotting.plot_flat_universe import plot_flat_universe
import run.flat_universe.run_flat_universe as flat
from science.populations.universes.flat_curves import is_super_earth
import run.hwo.run_hwo as hwo
import run.kepler.run_kepler as kepler
import run.lifesim.run_lifesim as lifesim
import run.rv.run_rv as rv
import run.tess.run_tess as tess
from science.populations.ppop import PPop
import sim

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
        normalized = [
            "" if value is None or (isinstance(value, float) and np.isnan(value))
            else str(value)
            for value in s
        ]
        return pd.Series(normalized, index=s.index)
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
        distance_range = list(SMALL_DIST_RANGE)
        self.Dist_range = distance_range


@pytest.fixture(scope='session')
def sandbox(tmp_path_factory):
    """Point every pipeline and plotter at a temp dir and at the small star catalog."""
    root = tmp_path_factory.mktemp('results')
    with pytest.MonkeyPatch.context() as mp:
        for name, (module, dir_attrs, _) in PPOP_PIPELINES.items():
            mp.setattr(module, 'PPop', SmallPPop)
            for attr in dir_attrs:
                directory = str(root / name)
                mp.setattr(module, attr, directory)
                if getattr(module.main, "keywords", {}).get("catalog_dir") is not None:
                    mp.setitem(module.main.keywords, "catalog_dir", directory)
        mp.setattr(flat, 'FLAT_CACHE_DIR', root / 'flat_universe')
        (root / 'flat_universe').mkdir()
        mp.setattr(base_plotter, 'PLOTS_DIR', str(root / 'figures'))
        mp.setattr(likelihood_ratio_plotter, 'OUT_DIR', str(root / 'likelihood_ratio'))
        mp.setattr(sim, 'LOGGING', str(root / 'logs'))
        yield root


@pytest.fixture(scope='session', params=list(PPOP_PIPELINES))
def ppop_run(request, sandbox):
    """One run_sim call per pipeline (two universes, with plots), shared by the tests below."""
    name = request.param
    module, _, detected_col = PPOP_PIPELINES[name]
    df = sim.run_sim(func=module.main, name=name, parallel=False, nruns=NRUNS,
                                star_catalog='Gaia', run_anew=True, plot=True)
    return name, module, detected_col, df


def test_ppop_pipeline_runs(ppop_run):
    name, _, detected_col, df = ppop_run
    assert sorted(df['run'].unique()) == list(NRUNS)
    missing = (PLANET_COLUMNS | {detected_col}) - set(df.columns)
    assert not missing, f'{name} output lacks {sorted(missing)}'
    assert df['radius_p'].gt(0).all() and df['mass_p'].gt(0).all()


def test_ppop_pipeline_detects(ppop_run):
    name, _, detected_col, df = ppop_run
    detected = df[detected_col].astype(bool)
    detected_count = int(detected.sum())
    assert detected_count > 0, f'{name} detected no planets'


def test_ppop_pipeline_plots(ppop_run, sandbox):
    name = ppop_run[0]
    pngs = list((sandbox / 'figures').glob(f'{name}_{len(NRUNS)}_Gaia/**/*.png'))
    assert pngs, f'plot_all wrote no figures for {name}'


def test_ppop_pipeline_reloads_saved_catalogs(ppop_run):
    name, module, _, df = ppop_run
    reloaded = module.main(parallel=False, nruns=NRUNS, star_catalog='Gaia', run_anew=False)
    got, expected = _csv_comparable(reloaded, df.drop(columns='radius_bin'))
    pd.testing.assert_frame_equal(got, expected, check_dtype=False)


def test_ppop_pipeline_is_seeded(ppop_run):
    name, module, _, df = ppop_run
    again = module.main(parallel=False, nruns=NRUNS[1:], star_catalog='Gaia', run_anew=True)
    got, expected = _csv_comparable(again, df[df['run'] == NRUNS[1]].drop(columns='radius_bin'))
    pd.testing.assert_frame_equal(got, expected)


def test_flat_universe(sandbox):
    n_planets = 20_000
    df = sim.run_sim(func=flat.main, name='flat_ab', parallel=False,
                                nruns=np.arange(1), run_anew=True, plot=False,
                                seed=0, n_planets=n_planets)

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
    df = df.drop(columns='radius_bin')
    pd.testing.assert_frame_equal(cached, df, check_dtype=False)
    pd.testing.assert_frame_equal(regenerated, df, check_dtype=False)

    plot_flat_universe(df)
    assert list((sandbox / 'likelihood_ratio').glob('*.png'))


def test_kepler_on_nasa_pscomppars(sandbox):
    out_csv = sandbox / 'kepler_nasa.csv'
    df = kepler.run_nasa_pscomppars(output_csv=out_csv)
    assert out_csv.exists()
    assert len(df) > 100
    assert 0 < df['detected'].sum() < len(df)


def test_kepler_rejects_unknown_population():
    invalid_population = 'not_a_population'
    with pytest.raises(ValueError):
        kepler.main(population=invalid_population)


def test_run_sim_accepts_integer_nruns(sandbox):
    def fake_pipeline(**kwargs):
        runs = kwargs["nruns"]
        assert runs == [0, 1, 2]
        result = pd.DataFrame({"radius_p": [1.0, 2.0], "run": [0, 1]})
        return result

    df = sim.run_sim(func=fake_pipeline, name="fake", parallel=False, nruns=3, plot=False)

    assert list(df["radius_bin"].astype(str)) == ["<1.5", "1.5–3.0"]

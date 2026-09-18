from tools.exoplanet_catalog import load_and_filter_exoplanets
from tools.paths import EXOPLANETS_2026_CSV
from science.telescopes.hwo.detection_model import HWOData
import numpy as np
import pandas as pd


def test_hwo_on_real_exoplanet_catalog():
    df = load_and_filter_exoplanets(EXOPLANETS_2026_CSV)
    n_in = len(df)

    hwo_data = HWOData(df)
    hwo_data.determine_detectable()
    out = hwo_data.catalog

    assert len(out) == n_in
    for col in ('detected_best', 'detected_worst'):
        assert out[col].dtype == bool
        assert 0 < out[col].sum() < len(out)
    # The worst case cannot detect a planet the best case misses.
    assert not (out['detected_worst'] & ~out['detected_best']).any()


def test_kepler_observed_depth_falls_back_to_model():
    from science.telescopes.kepler.detection_model import KeplerData

    catalog = pd.DataFrame({
        'radius_p': [1.0, 2.0], 'radius_s': [1.0, 1.0],
        'observed_transit_depth_ppm': [123.0, np.nan],
    })
    detector = KeplerData(catalog, source='nasa', validate_for_detection=False,
                          estimate_missing_semimajor_axis=False)
    np.testing.assert_allclose(detector.calc_transit_depth_ppm(),
                               [123.0, (2.0 / detector.R_SUN_IN_R_EARTH) ** 2 * 1e6])
    assert detector.catalog.transit_depth_source.tolist() == ['observed', 'model_Rp_Rstar']


def test_tic_enrichment_only_fills_missing_values(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace
    from astropy.table import Table
    from science.telescopes.tess.detection_model import TESSData

    matches = Table({'ID': [42], 'Tmag': [9.0], 'rad': [0.7],
                     'mass': [0.6], 'Teff': [np.nan]})
    mast = ModuleType('astroquery.mast')
    mast.Catalogs = SimpleNamespace(query_region=lambda *args, **kwargs: matches)
    monkeypatch.setitem(sys.modules, 'astroquery', ModuleType('astroquery'))
    monkeypatch.setitem(sys.modules, 'astroquery.mast', mast)
    detector = TESSData.__new__(TESSData)
    detector.mast_max_rows = 1
    detector.catalog = pd.DataFrame({'ra': [10.0], 'dec': [20.0], 'tess_tmag': [8.0],
                                     'radius_s': [np.nan], 'mass_s': [np.nan], 'teff_s': [np.nan]})
    detector._enrich_from_tic()
    row = detector.catalog.iloc[0]
    assert row.tess_tmag == 8.0
    assert row.radius_s == 0.7
    assert row.mass_s == 0.6
    assert np.isnan(row.teff_s)
    assert row.tic_query_status == 'matched'


def test_rejection_plotter_inherits_initialization(tmp_path, monkeypatch):
    from plotting import base_plotter
    from plotting.plot_rejections import PlanetRejectionPlotter

    monkeypatch.setattr(base_plotter, 'PLOTS_DIR', str(tmp_path))
    catalog = pd.DataFrame({'detected_best': [True, False], 'detected_worst': [False, False]})
    plotter = PlanetRejectionPlotter(catalog)
    assert plotter.name == 'HWO'
    best, worst = plotter._get_detection_masks()
    assert best.tolist() == [True, False]
    assert worst.tolist() == [False, False]

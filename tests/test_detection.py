from tools.exoplanet_catalog import load_and_filter_exoplanets
from tools.paths import EXOPLANETS_2026_CSV
from telescopes.hwo.detection_model import HWOData


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

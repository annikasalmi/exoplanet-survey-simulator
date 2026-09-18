"""Flat-universe priors, analysis windows, and the baseline quickstart."""

import runpy
import sys
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

from science.populations.universes.flat_baseline import DEFAULTS, flat_nonphysical
from science.populations.universes.flat_curves import flat_radii_curves, is_super_earth
from tools import paths


def test_baseline_radius_prior_and_mass_relation():
    cat = flat_nonphysical(20_000, seed=4, mass_scatter_dex=0)
    assert len(cat) == 20_000
    assert cat.radius_p.between(0, 12).all()
    counts, _ = np.histogram(cat.radius_p, bins=np.arange(13))
    assert np.all(np.abs(counts - len(cat) / 12) < 200)
    mass = np.maximum((cat.radius_p / 1.03) ** (1 / 0.29), 0.1)
    np.testing.assert_allclose(cat.mass_p, mass)


def test_paper_window_preserves_previous_catalog():
    kw = dict(radius_lims=DEFAULTS["radius_lims"], mass_lims=DEFAULTS["mass_lims"])
    actual = flat_nonphysical(2000, seed=75, **kw)
    # Seeded values from the paper's original sampler, including its mass cutoff.
    assert len(actual) == 1847
    np.testing.assert_allclose(actual[["radius_p", "mass_p", "p_orb"]].head(3), [
        [0.8625958585945364, 0.4721644993397367, 6231.466213497824],
        [1.629225769291016, 5.34401147852668, 93.77770872502197],
        [1.6633639898369954, 5.975752238127815, 479.9998392704545],
    ])
    pd.testing.assert_frame_equal(actual, flat_nonphysical(2000, seed=75, **kw))


def test_curve_based_a_is_b_without_super_earths():
    b = flat_radii_curves(2000, seed=0)
    a = flat_radii_curves(2000, seed=0, variant="only_subneptunes")
    assert 0 < len(a) < len(b) <= 2000
    assert b.radius_p.between(*DEFAULTS["radius_lims"]).all()
    assert b.mass_p.between(*DEFAULTS["mass_lims"]).all()
    expected = b[~is_super_earth(b.mass_p, b.radius_p)].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, expected)

@pytest.mark.parametrize(
    "universe,variant",
    [("flat_nonphysical", "superearths_supneptunes"),
     ("flat_radii_curves", "only_subneptunes"),
     ("flat_radii_curves", "superearths_supneptunes")],
)
def test_flat_quickstart_writes_catalog_and_plots(tmp_path, monkeypatch, universe, variant):
    import plotting.plot_population as plotter

    monkeypatch.setattr(paths, "CATALOGS_DIR", str(tmp_path / "catalogs"))
    monkeypatch.setattr(plotter, "PLOTS_DIR", str(tmp_path / "plots"))
    monkeypatch.setattr(sys, "argv", ["sim.py", "--universe", universe, "--variant", variant,
                                     "--n-planets", "2000"])
    runpy.run_path(str(Path(paths.REPO_ROOT) / "sim.py"), run_name="__main__")

    output, = (tmp_path / "catalogs" / universe).glob("*.csv")
    cat = pd.read_csv(output)
    assert 0 < len(cat) <= 2000
    detected = cat[["kepler_detected", "tess_detected", "rv_detected"]]
    assert detected.any().all()
    figures = list((tmp_path / "plots").rglob("*.png"))
    assert {path.name for path in figures} == {"population.png", "detections.png"}
    for path in figures:
        pixels = mpimg.imread(path)
        assert pixels.shape[0] > 100 and pixels.shape[1] > 100
        assert pixels.std() > 0.01

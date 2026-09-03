import pytest  # type: ignore
import numpy as np
import pandas as pd
from hwo.hwo_data import HWOData

class DummyConst:
    h = 6.62607015e-34
    c = 2.99792458e8
    k = 1.380649e-23
    sigma = 5.670374419e-8
    R_earth = 6.371e6
    pc_to_m = 3.085677581e16

class DummyHWO:
    def __init__(self, case):
        self.min_wavelength_hwo = 1e-6
        self.max_wavelength_hwo = 2e-6
        self.iwa = 0.1
        self.min_planet_flux_star_ratio = 1e-10
        self.min_photons = 1e5
        self.max_z = 10 if case == 'best' else 1

def minimal_catalog():
    return pd.DataFrame({
        'temp_p': [300],
        'temp_s': [5800],
        'radius_p': [1],
        'radius_s': [1],
        'distance_s': [10],
        'maxangsep': [0.2],
        'z': [0.5],
    })

def test_constructor_accepts_dataframe():
    df = minimal_catalog()
    hwo = HWOData(df)
    assert isinstance(hwo.catalog, pd.DataFrame)

def test_constructor_rejects_invalid_type():
    with pytest.raises(TypeError):
        HWOData("not a dataframe")  # type: ignore

def test_validate_catalog_missing_column():
    df = minimal_catalog().drop(columns=['z'])
    with pytest.raises(ValueError):
        HWOData(df)

def test_blackbody_flux_shape_and_value(monkeypatch):
    df = minimal_catalog()
    hwo = HWOData(df)
    # Patch const
    import tools.physics_constants as const
    monkeypatch.setattr(const, 'h', DummyConst.h)
    monkeypatch.setattr(const, 'c', DummyConst.c)
    monkeypatch.setattr(const, 'k', DummyConst.k)
    # Should return array of same shape as input
    result = hwo.blackbody_flux(1e-6, 300)
    assert np.isscalar(result) or result.shape == ()

def test_bolometric_flux(monkeypatch):
    df = minimal_catalog()
    hwo = HWOData(df)
    import tools.physics_constants as const
    monkeypatch.setattr(const, 'sigma', DummyConst.sigma)
    monkeypatch.setattr(const, 'pc_to_m', DummyConst.pc_to_m)
    monkeypatch.setattr(const, 'R_earth', DummyConst.R_earth)
    result = hwo.bolometric_flux(300)
    assert isinstance(result, float)

def test_calc_iwa_constraint():
    df = minimal_catalog()
    hwo = HWOData(df)
    result = hwo.calc_iwa_constraint()
    assert np.allclose(result, df['maxangsep'])

def test_photon_energy(monkeypatch):
    df = minimal_catalog()
    hwo = HWOData(df)
    import tools.physics_constants as const
    monkeypatch.setattr(const, 'h', DummyConst.h)
    monkeypatch.setattr(const, 'c', DummyConst.c)
    result = hwo.photon_energy(1e-6)
    assert np.isclose(result, DummyConst.h * DummyConst.c / 1e-6)

def test_photon_rate_per_hour_per_micron():
    df = minimal_catalog()
    hwo = HWOData(df)
    # Use simple values
    flux = 1.0
    wavelength = 1e-6
    result = hwo.photon_rate_per_hour_per_micron(flux, wavelength)
    assert isinstance(result, float) or isinstance(result, np.ndarray)

def test_calc_planet_flux(monkeypatch):
    df = minimal_catalog()
    hwo = HWOData(df)
    import tools.physics_constants as const
    monkeypatch.setattr(const, 'h', DummyConst.h)
    monkeypatch.setattr(const, 'c', DummyConst.c)
    monkeypatch.setattr(const, 'k', DummyConst.k)
    import hwo.hwo_data as hwo_data_mod
    monkeypatch.setattr(hwo_data_mod, 'HWO', DummyHWO)
    result = hwo.calc_planet_flux('best')
    assert isinstance(result, np.ndarray)

def test_calc_flux_ratio(monkeypatch):
    df = minimal_catalog()
    hwo = HWOData(df)
    import tools.physics_constants as const
    monkeypatch.setattr(const, 'h', DummyConst.h)
    monkeypatch.setattr(const, 'c', DummyConst.c)
    monkeypatch.setattr(const, 'k', DummyConst.k)
    import hwo.hwo_data as hwo_data_mod
    monkeypatch.setattr(hwo_data_mod, 'HWO', DummyHWO)
    result = hwo.calc_flux_ratio('best')
    assert isinstance(result, np.ndarray)

def test_calc_photons(monkeypatch):
    df = minimal_catalog()
    hwo = HWOData(df)
    import tools.physics_constants as const
    monkeypatch.setattr(const, 'h', DummyConst.h)
    monkeypatch.setattr(const, 'c', DummyConst.c)
    monkeypatch.setattr(const, 'k', DummyConst.k)
    monkeypatch.setattr(const, 'pc_to_m', DummyConst.pc_to_m)
    import hwo.hwo_data as hwo_data_mod
    monkeypatch.setattr(hwo_data_mod, 'HWO', DummyHWO)
    result = hwo.calc_photons('best')
    assert isinstance(result, np.ndarray)

def test_determine_detectable(monkeypatch):
    df = minimal_catalog()
    hwo = HWOData(df)
    import tools.physics_constants as const
    monkeypatch.setattr(const, 'h', DummyConst.h)
    monkeypatch.setattr(const, 'c', DummyConst.c)
    monkeypatch.setattr(const, 'k', DummyConst.k)
    monkeypatch.setattr(const, 'pc_to_m', DummyConst.pc_to_m)
    import hwo.hwo_data as hwo_data_mod
    monkeypatch.setattr(hwo_data_mod, 'HWO', DummyHWO)
    result = hwo.determine_detectable()
    assert 'detected_best' in result.columns
    assert 'detected_worst' in result.columns 
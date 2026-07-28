"""
flat_detect.py — run Kepler / TESS / RV detectors on a flat catalogue.

Loads the three detector modules by file path (so the heavy lifesim package
__init__, which pulls in a Qt GUI dependency, is never triggered — their own
`from lifesim.core.data import Data` is wrapped in try/except). Mirrors the
importlib pattern already used in plot/script plots/57_rv_mr_detection.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_kep = _load("kepler_data", "lifesim/core/kepler_data.py")
_tess = _load("tess_data", "lifesim/core/tess_data.py")
_rv = _load("rv_data", "lifesim/core/rv_data.py")

KeplerData = _kep.KeplerData
TESSData = _tess.TESSData
RVData = _rv.RVData


def run_kepler(catalog: pd.DataFrame) -> pd.DataFrame:
    return KeplerData(catalog.copy(), source="ppop").determine_detectable()


def run_tess(catalog: pd.DataFrame) -> pd.DataFrame:
    return TESSData(catalog.copy(), source="ppop", use_tesspoint=False).determine_detectable()


def run_rv(catalog: pd.DataFrame, instrument: str = "HARPS") -> pd.DataFrame:
    return RVData(catalog.copy(), source="ppop", instrument=instrument).determine_detectable()


def run_rv_best(catalog: pd.DataFrame, mag_target: float = 12.0) -> pd.DataFrame:
    """Per-planet BEST of HARPS (optical V) and NIRPS (NIR J) — the realistic
    'pick the right spectrograph per target' case used in the 3x4 reference figure.

    Returns the HARPS frame with:
        detected  = HARPS_detected OR NIRPS_detected
        rv_is_target = host reachable (band-mag <= mag_target) in EITHER band
    so callers can use rv_is_target as the RV denominator (matches the
    'band-mag <= 12' RV-target cut in script 50 / the 3x4 figure).
    """
    h = run_rv(catalog, "HARPS")
    n = run_rv(catalog, "NIRPS")
    best = h.copy()
    best["detected"] = h["detected"].astype(bool) | n["detected"].astype(bool)
    h_mag = pd.to_numeric(h["rv_mag"], errors="coerce")
    n_mag = pd.to_numeric(n["rv_mag"], errors="coerce")
    best["rv_is_target"] = (h_mag <= mag_target) | (n_mag <= mag_target)
    best["rv_best_band"] = pd.Series(
        ["NIRPS" if (nd and not hd) else "HARPS"
         for hd, nd in zip(h["detected"].astype(bool), n["detected"].astype(bool))],
        index=best.index,
    )
    return best


def run_all(catalog: pd.DataFrame, rv_instrument: str = "HARPS") -> dict[str, pd.DataFrame]:
    """Return {'Kepler':df, 'TESS':df, 'RV':df}, each with a 'detected' column."""
    return {
        "Kepler": run_kepler(catalog),
        "TESS": run_tess(catalog),
        "RV": run_rv(catalog, rv_instrument),
    }

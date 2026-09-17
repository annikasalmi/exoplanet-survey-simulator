"""Run the Kepler / TESS / RV detectors on a flat catalog."""

from __future__ import annotations

import pandas as pd

from science.telescopes.kepler.detection_model import KeplerData
from science.telescopes.tess.detection_model import TESSData
from science.telescopes.rv.detection_model import RVData


def run_kepler(catalog: pd.DataFrame) -> pd.DataFrame:
    return KeplerData(catalog.copy(), source="ppop").determine_detectable()


def run_tess(catalog: pd.DataFrame) -> pd.DataFrame:
    return TESSData(catalog.copy(), source="ppop", use_cdpp_tables=False).determine_detectable()


def run_rv(catalog: pd.DataFrame, instrument: str = "HARPS") -> pd.DataFrame:
    return RVData(catalog.copy(), source="ppop", instrument=instrument).determine_detectable()


def run_rv_best(catalog: pd.DataFrame, mag_target: float = 12.0) -> pd.DataFrame:
    """Per-planet best of HARPS (V band) and NIRPS (J band). Returns the HARPS frame with detected =
    either instrument, and rv_is_target = host bright enough in either band (the RV denominator).
    """
    h = run_rv(catalog, "HARPS")
    n = run_rv(catalog, "NIRPS")
    best = h.copy()
    best["detected"] = h["detected"].astype(bool) | n["detected"].astype(bool)
    # RVData has no best/worst split (both copy `detected`), so keep them on the
    # combined value rather than HARPS alone.
    best["detected_best"] = best["detected"]
    best["detected_worst"] = best["detected"]
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

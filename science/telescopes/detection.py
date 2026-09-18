"""Combine the optical and near-infrared RV selection models."""

from __future__ import annotations

import pandas as pd

from science.telescopes.rv.detection_model import RVData


def run_rv_best(catalog: pd.DataFrame, mag_target: float = 12.0) -> pd.DataFrame:
    """Per-planet best of HARPS (V band) and NIRPS (J band). Returns the HARPS frame with detected =
    either instrument, and rv_is_target = host bright enough in either band (the RV denominator).
    """
    h = RVData(catalog.copy(), source="ppop", instrument="HARPS").determine_detectable()
    n = RVData(catalog.copy(), source="ppop", instrument="NIRPS").determine_detectable()
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

"""Catalog normalization helpers for the RV toy detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from science.physics import infer_stellar_type

try:
    from lifesim.core.data import Data
except Exception:  # lets this file import outside the lifesim environment
    Data = None


def as_dataframe(data) -> pd.DataFrame:
    if Data is not None and isinstance(data, Data):
        return pd.DataFrame(data.catalog).copy()
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data).copy()


def infer_source(catalog: pd.DataFrame, source: str) -> str:
    source = str(source).lower().strip()
    if source != "auto":
        return source
    cols = set(catalog.columns)
    if {"pl_name", "pl_bmasse"}.issubset(cols) or {"pl_name", "pl_msinie"}.issubset(cols):
        return "pscomppars"
    if {"kepoi_name", "koi_prad"}.issubset(cols):
        return "koi"
    return "ppop"


def standardize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    rename = {
        "pl_name": "planet_name", "hostname": "host_name",
        "pl_orbper": "p_orb", "pl_orbsmax": "semimajor_p", "pl_orbincl": "inc_p",
        "pl_orbeccen": "ecc_p", "pl_rade": "radius_p",
        "pl_bmasse": "mass_p", "pl_msinie": "msini_p",
        "pl_rvamp": "rv_amp_obs", "pl_insol": "flux_p",
        "st_rad": "radius_s", "st_mass": "mass_s", "st_teff": "teff_s",
        "st_lum": "st_lum_log10", "sy_dist": "distance_s",
        "sy_vmag": "vmag", "sy_gaiamag": "gaiamag",
        "kepoi_name": "planet_name", "koi_period": "p_orb", "koi_prad": "radius_p",
        "koi_incl": "inc_p", "koi_srad": "radius_s", "koi_smass": "mass_s",
        "koi_steff": "teff_s", "koi_insol": "flux_p",
        "luminosity_s": "l_sun", "temp_s": "teff_s", "insolation": "flux_p",
        "Vmag": "vmag", "gaia_g_mag": "gaiamag",
    }
    for src_col, dst_col in rename.items():
        if src_col in df.columns and dst_col not in df.columns:
            df = df.rename(columns={src_col: dst_col})

    if "st_lum_log10" in df.columns and "l_sun" not in df.columns:
        df["l_sun"] = 10 ** pd.to_numeric(df["st_lum_log10"], errors="coerce")

    for col in ["mass_p", "msini_p", "radius_p", "p_orb", "semimajor_p", "inc_p",
                "ecc_p", "radius_s", "mass_s", "teff_s", "l_sun", "distance_s",
                "flux_p", "vmag", "gaiamag", "rv_amp_obs"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["dataset_source"] = {
        "pscomppars": "NASA_PSCompPars", "nasa": "NASA_PSCompPars", "koi": "NASA_KOI",
    }.get(source, "P-Pop_simulated")
    return df


def add_basic_columns(catalog: pd.DataFrame) -> pd.DataFrame:
    catalog = catalog.copy()
    if "stype" not in catalog.columns:
        teff = catalog.get("teff_s", pd.Series(np.nan, index=catalog.index))
        catalog["stype"] = infer_stellar_type(teff)
    else:
        catalog["stype"] = (
            catalog["stype"].astype(str).str.strip().str.upper().str[0]
            .where(lambda s: s.isin(["A", "F", "G", "K", "M"]), "Unknown")
        )
    if "ecc_p" not in catalog.columns:
        catalog["ecc_p"] = 0.0
    catalog["ecc_p"] = pd.to_numeric(catalog["ecc_p"], errors="coerce").fillna(0.0).clip(0.0, 0.95)
    if "habitable" not in catalog.columns and "flux_p" in catalog.columns:
        flux = pd.to_numeric(catalog["flux_p"], errors="coerce")
        catalog["habitable"] = (flux >= 0.25) & (flux <= 2.0)
    return catalog


def validate_detection_columns(catalog: pd.DataFrame) -> None:
    required = ["mass_p", "p_orb"]
    missing = [c for c in required if c not in catalog.columns]
    if "mass_p" in missing and "msini_p" in catalog.columns:
        missing.remove("mass_p")
    if missing:
        raise ValueError(
            f"Missing required RV detection columns: {missing}. "
            "Need a planet mass (mass_p or msini_p) and orbital period (p_orb)."
        )


def prepare_catalog(data, source: str, *, validate_for_detection: bool) -> tuple[pd.DataFrame, str]:
    catalog = as_dataframe(data)
    source = infer_source(catalog, source)
    catalog = standardize(catalog, source)
    catalog = add_basic_columns(catalog)
    if validate_for_detection:
        validate_detection_columns(catalog)
    return catalog, source

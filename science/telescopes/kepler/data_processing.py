"""Catalog normalization helpers for the Kepler toy detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from science.physics import infer_stellar_type, semimajor_axis_from_period

try:
    from lifesim.core.data import Data
except Exception:  # lets this file import outside the lifesim environment
    Data = None


def as_dataframe(data) -> pd.DataFrame:
    if Data is not None and isinstance(data, Data):
        return pd.DataFrame(data.catalog)
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame(data)


def infer_source(catalog: pd.DataFrame, source: str) -> str:
    source = str(source).lower().strip()
    if source != "auto":
        return source

    cols = set(catalog.columns)
    if {"pl_name", "pl_rade", "pl_insol"}.issubset(cols):
        return "pscomppars"
    if {"kepoi_name", "koi_prad", "koi_insol"}.issubset(cols):
        return "koi"
    return "ppop"


def standardize_catalog_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Rename NASA / KOI / P-Pop columns to the shared detector names."""
    df = df.copy()
    source = str(source).lower().strip()

    ps_map = {
        "pl_name": "planet_name",
        "hostname": "host_name",
        "discoverymethod": "discovery_method",
        "disc_facility": "discovery_facility",
        "disc_telescope": "discovery_telescope",
        "tran_flag": "tran_flag",
        "pl_orbper": "p_orb",
        "pl_orbsmax": "semimajor_p",
        "pl_rade": "radius_p",
        "pl_bmasse": "mass_p",
        "pl_masse": "mass_p",
        "pl_insol": "flux_p",
        "pl_orbincl": "inc_p",
        "st_rad": "radius_s",
        "st_mass": "mass_s",
        "st_teff": "teff_s",
        "st_lum": "st_lum_log10",
        "sy_dist": "distance_s",
        "sy_kepmag": "kepmag",
        "sy_gaiamag": "gaiamag",
        "pl_trandep": "observed_transit_depth_percent",
        "pl_trandur": "observed_transit_duration_hr",
        "pl_bmasse_reflink": "mass_reference",
        "pl_rade_reflink": "radius_reference",
    }

    koi_map = {
        "kepid": "kepid",
        "kepoi_name": "planet_name",
        "kepler_name": "kepler_name",
        "koi_disposition": "koi_disposition",
        "koi_pdisposition": "koi_pdisposition",
        "koi_score": "koi_score",
        "koi_period": "p_orb",
        "koi_prad": "radius_p",
        "koi_sma": "semimajor_p",
        "koi_insol": "flux_p",
        "koi_depth": "observed_transit_depth_ppm",
        "koi_duration": "observed_transit_duration_hr",
        "koi_ror": "observed_radius_ratio",
        "koi_incl": "inc_p",
        "koi_steff": "teff_s",
        "koi_srad": "radius_s",
        "koi_smass": "mass_s",
        "koi_kepmag": "kepmag",
    }

    generic_map = {
        "luminosity_s": "l_sun",
        "temp_s": "teff_s",
        "star_teff": "teff_s",
        "pl_insol": "flux_p",
        "insolation": "flux_p",
    }

    if source in {"pscomppars", "ps", "nasa"}:
        rename_map = {k: v for k, v in ps_map.items() if k in df.columns and v not in df.columns}
        df = df.rename(columns=rename_map)
        df["dataset_source"] = "NASA_PSCompPars"
    elif source == "koi":
        rename_map = {k: v for k, v in koi_map.items() if k in df.columns and v not in df.columns}
        df = df.rename(columns=rename_map)
        df["dataset_source"] = "NASA_KOI"
    elif source == "ppop":
        df["dataset_source"] = "P-Pop_simulated"
    else:
        raise ValueError("source must be one of: ppop, pscomppars, nasa, ps, koi")

    rename_map = {k: v for k, v in generic_map.items() if k in df.columns and v not in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    if "st_lum_log10" in df.columns and "l_sun" not in df.columns:
        st_lum = pd.to_numeric(df["st_lum_log10"], errors="coerce")
        df["l_sun"] = 10 ** st_lum
        df["l_sun_source"] = "10**st_lum_from_PSCompPars"

    numeric_cols = [
        "mass_p", "radius_p", "flux_p", "radius_s", "mass_s", "teff_s",
        "semimajor_p", "p_orb", "inc_p", "kepmag", "gaiamag", "distance_s",
        "observed_transit_depth_ppm", "observed_transit_depth_percent",
        "observed_transit_duration_hr", "l_sun",
        "rrmscdpp01p5", "rrmscdpp02p0", "rrmscdpp02p5", "rrmscdpp03p0",
        "rrmscdpp03p5", "rrmscdpp04p5", "rrmscdpp05p0", "rrmscdpp06p0",
        "rrmscdpp07p5", "rrmscdpp09p0", "rrmscdpp10p5", "rrmscdpp12p0",
        "rrmscdpp12p5", "rrmscdpp15p0", "dataspan", "dutycycle",
        "koi_max_sngle_ev", "koi_max_mult_ev", "koi_model_snr", "koi_num_transits",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "observed_transit_depth_percent" in df.columns:
        depth_percent = pd.to_numeric(df["observed_transit_depth_percent"], errors="coerce")
        depth_ppm_from_percent = depth_percent * 1e4
        if "observed_transit_depth_ppm" in df.columns:
            existing = pd.to_numeric(df["observed_transit_depth_ppm"], errors="coerce")
            df["observed_transit_depth_ppm"] = existing.fillna(depth_ppm_from_percent)
        else:
            df["observed_transit_depth_ppm"] = depth_ppm_from_percent

    return df


def add_basic_helper_columns(catalog: pd.DataFrame) -> pd.DataFrame:
    catalog = catalog.copy()
    if "stype" not in catalog.columns:
        if "teff_s" in catalog.columns:
            catalog["stype"] = infer_stellar_type(catalog["teff_s"])
        else:
            catalog["stype"] = "Unknown"

    if "habitable" not in catalog.columns:
        if "flux_p" in catalog.columns:
            flux = pd.to_numeric(catalog["flux_p"], errors="coerce")
            catalog["habitable"] = (flux >= 0.25) & (flux <= 2.0)
        else:
            catalog["habitable"] = False

    if "l_sun" in catalog.columns and "luminosity_s" not in catalog.columns:
        catalog["luminosity_s"] = catalog["l_sun"]
    if "teff_s" in catalog.columns and "temp_s" not in catalog.columns:
        catalog["temp_s"] = catalog["teff_s"]
    return catalog


def estimate_missing_semimajor_axis_if_possible(catalog: pd.DataFrame) -> pd.DataFrame:
    catalog = catalog.copy()
    if "semimajor_p" not in catalog.columns:
        catalog["semimajor_p"] = np.nan

    missing_a = catalog["semimajor_p"].isna()
    if not missing_a.any() or "p_orb" not in catalog.columns:
        return catalog

    p_days = pd.to_numeric(catalog["p_orb"], errors="coerce")
    if "mass_s" in catalog.columns:
        mstar = pd.to_numeric(catalog["mass_s"], errors="coerce").fillna(1.0)
        mass_source = "mass_s"
    else:
        mstar = pd.Series(1.0, index=catalog.index)
        mass_source = "assumed_1Msun"

    a_est = pd.Series(semimajor_axis_from_period(p_days, mstar), index=catalog.index)
    catalog.loc[missing_a, "semimajor_p"] = a_est.loc[missing_a]
    catalog["semimajor_p_source"] = np.where(
        missing_a,
        f"estimated_from_period_and_{mass_source}",
        "catalog",
    )
    return catalog


def validate_detection_columns(catalog: pd.DataFrame) -> None:
    required = ["radius_p", "radius_s", "semimajor_p", "p_orb"]
    missing = [col for col in required if col not in catalog.columns]
    if missing:
        raise ValueError(
            "Missing required columns for Kepler detection: "
            + str(missing)
            + "\nIf this is NASA PSCompPars and you only want plotting, use "
            + "KeplerData(df, source='pscomppars', validate_for_detection=False)."
        )

    bad = []
    for col in required:
        values = pd.to_numeric(catalog[col], errors="coerce")
        if values.notna().sum() == 0:
            bad.append(col)
    if bad:
        raise ValueError(f"Detection columns exist but are entirely NaN: {bad}")


def prepare_catalog(
    data,
    source: str,
    *,
    estimate_missing_semimajor_axis: bool,
    validate_for_detection: bool,
) -> tuple[pd.DataFrame, str, bool]:
    catalog = as_dataframe(data)
    if catalog is None or not isinstance(catalog, pd.DataFrame):
        raise ValueError("Catalog must be a valid pandas DataFrame.")

    source = infer_source(catalog, source)
    catalog = standardize_catalog_columns(catalog, source)
    nasa_like_source = source in {"pscomppars", "ps", "nasa", "koi"}
    catalog = add_basic_helper_columns(catalog)
    if estimate_missing_semimajor_axis:
        catalog = estimate_missing_semimajor_axis_if_possible(catalog)
    if validate_for_detection:
        validate_detection_columns(catalog)
    return catalog, source, nasa_like_source

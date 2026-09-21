"""Catalog ingestion, normalization, and scientific sample selection."""

from __future__ import annotations

from itertools import takewhile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd

from science.physics import (
    add_stellar_type,
    blackbody_spectral_radiance,
    radius_on_curve,
)
from tools import physics_constants as const


COMPARISON_PARAMETER_BOX = {
    "r_lo": 0.5, "r_hi": 2.2, "m_lo": 0.1, "m_hi": 12.0,
    "f_lo": 1e-2, "f_hi": 1e4,
}
FGKM_TEMPERATURE_BOUNDS = (0.0, 7500.0)
DEFAULT_ROCKY_CATALOG_CUTS = {
    "exclude_mass_limits": True, "exclude_radius_limits": True,
    "require_two_sided_mass": True, "require_two_sided_radius": True,
    "max_mass_relative_uncertainty": 0.25, "max_radius_relative_uncertainty": 0.08,
}
FACILITY_BY_INSTRUMENT = {
    "KEPLER": "Kepler",
    "TESS": "Transiting Exoplanet Survey Satellite (TESS)",
}
DIRECT_IMAGING_WAVELENGTH_M = {"LIFE": 18.5e-6, "HWO": 2.5e-6}
NASA_TAP_SYNC_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
NASA_ROCKY_QUERY = """
SELECT pl_name, hostname, discoverymethod, disc_facility, tran_flag,
       pl_insol, pl_insolerr1, pl_insolerr2, pl_insollim,
       pl_rade, pl_radeerr1, pl_radeerr2, pl_radelim, pl_rade_reflink,
       pl_bmasse, pl_bmasseerr1, pl_bmasseerr2, pl_bmasselim,
       pl_bmassprov, pl_bmasse_reflink,
       st_teff, st_rad, st_mass, st_lum, sy_dist
FROM pscomppars
WHERE tran_flag = 1
  AND pl_rade IS NOT NULL
  AND pl_bmasse IS NOT NULL
  AND pl_insol IS NOT NULL
"""
_NASA_COLUMN_MAP = {
    "pl_name": "planet_name", "hostname": "host_name",
    "discoverymethod": "discovery_method", "disc_facility": "discovery_facility",
    "tran_flag": "transit_flag", "pl_bmassprov": "mass_provenance",
    "pl_bmasselim": "mass_limit_flag", "pl_radelim": "radius_limit_flag",
    "pl_insollim": "insolation_limit_flag", "pl_bmasse_reflink": "mass_reference",
    "pl_rade_reflink": "radius_reference", "st_rad": "stellar_radius",
    "st_mass": "stellar_mass", "st_lum": "stellar_luminosity",
    "sy_dist": "system_distance",
    "pl_bmasse": "mass", "pl_rade": "radius", "pl_insol": "insolation",
    "st_teff": "effective_temperature", "pl_bmasseerr1": "mass_error_plus",
    "pl_bmasseerr2": "mass_error_minus", "pl_radeerr1": "radius_error_plus",
    "pl_radeerr2": "radius_error_minus",
    "pl_insolerr1": "insolation_error_plus",
    "pl_insolerr2": "insolation_error_minus",
}
_NASA_NUMERIC_COLUMNS = {
    "transit_flag", "mass_limit_flag", "radius_limit_flag", "insolation_limit_flag",
    "stellar_radius", "stellar_mass", "stellar_luminosity", "system_distance",
    "mass", "radius", "insolation", "effective_temperature", "mass_error_plus",
    "mass_error_minus", "radius_error_plus", "radius_error_minus",
    "insolation_error_plus", "insolation_error_minus",
}


def read_nasa_csv(path, **kwargs) -> pd.DataFrame:
    """Read a NASA Archive/ExoFOP export without truncating HTML entities."""
    with open(path, encoding="utf-8", errors="replace") as stream:
        header_rows = sum(
            1 for _ in takewhile(lambda line: line.startswith("#"), stream)
        )
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(path, skiprows=header_rows, **kwargs)


def nasa_tap_url(query: str) -> str:
    """NASA Exoplanet Archive TAP CSV URL for a SQL query."""
    encoded_query = quote(query)
    url = f"{NASA_TAP_SYNC_URL}?query={encoded_query}&format=csv"
    return url


def parameter_box_mask(box, mass, radius, insolation=None):
    """Select finite mass/radius values inside a named parameter box."""
    mass = np.asarray(mass, dtype=float)
    radius = np.asarray(radius, dtype=float)
    selected = (
        np.isfinite(mass) & np.isfinite(radius)
        & (mass >= box["m_lo"]) & (mass <= box["m_hi"])
        & (radius >= box["r_lo"]) & (radius <= box["r_hi"])
    )
    if insolation is not None:
        insolation = np.asarray(insolation, dtype=float)
        selected &= (
            np.isfinite(insolation)
            & (insolation >= box["f_lo"]) & (insolation <= box["f_hi"])
        )
    return selected


def load_measured_planets(
    path,
    *,
    mass_bounds=(COMPARISON_PARAMETER_BOX["m_lo"], COMPARISON_PARAMETER_BOX["m_hi"]),
    radius_bounds=(COMPARISON_PARAMETER_BOX["r_lo"], COMPARISON_PARAMETER_BOX["r_hi"]),
    insolation_bounds=(
        COMPARISON_PARAMETER_BOX["f_lo"], COMPARISON_PARAMETER_BOX["f_hi"]
    ),
    temperature_bounds=None,
    max_relative_error=None,
    missing_relative_error=None,
):
    """Load measured NASA masses with normalized values and uncertainties."""
    raw = read_nasa_csv(path)
    frame = pd.DataFrame(index=raw.index)
    for source, target in _NASA_COLUMN_MAP.items():
        if source not in raw:
            frame[target] = np.nan if target in _NASA_NUMERIC_COLUMNS else ""
        elif target in _NASA_NUMERIC_COLUMNS:
            frame[target] = pd.to_numeric(raw[source], errors="coerce")
        else:
            frame[target] = raw[source]
        if target.endswith(("_error_plus", "_error_minus")):
            frame[target] = frame[target].abs()

    provenance = frame["mass_provenance"].astype(str)
    measured = provenance.str.contains("Mass|Msini", case=False, na=False)
    measured &= ~provenance.str.contains("Calc", case=False, na=False)
    selected = (
        measured.to_numpy(bool)
        & frame["mass"].between(*mass_bounds).to_numpy(bool)
        & frame["radius"].between(*radius_bounds).to_numpy(bool)
    )
    if insolation_bounds is not None:
        selected &= frame["insolation"].between(*insolation_bounds).to_numpy(bool)
    if temperature_bounds is not None:
        selected &= frame["effective_temperature"].between(
            *temperature_bounds
        ).to_numpy(bool)

    frame["mass_error"] = frame[["mass_error_plus", "mass_error_minus"]].max(axis=1)
    frame["radius_error"] = frame[["radius_error_plus", "radius_error_minus"]].max(axis=1)
    if max_relative_error is not None:
        selected &= (
            (frame["mass_error"] / frame["mass"] <= max_relative_error["mass"])
            & (frame["radius_error"] / frame["radius"] <= max_relative_error["radius"])
        ).to_numpy(bool)

    frame = frame.loc[selected].reset_index(drop=True)
    if missing_relative_error is not None:
        for quantity in ("mass", "radius"):
            for suffix in ("error_plus", "error_minus", "error"):
                column = f"{quantity}_{suffix}"
                valid = np.isfinite(frame[column]) & (frame[column] > 0)
                frame[column] = frame[column].where(
                    valid, missing_relative_error[quantity] * frame[quantity]
                )
    return frame


def load_nasa_rocky_source(cache_path, *, redownload=False, download_if_missing=True):
    """Load and normalize the cached rocky query, downloading it when permitted."""
    cache_path = Path(cache_path)
    if redownload or not cache_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"NASA cache missing: {cache_path}")
        frame = pd.read_csv(nasa_tap_url(NASA_ROCKY_QUERY))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path, index=False)
    return load_measured_planets(
        cache_path,
        mass_bounds=(0.0, np.inf),
        radius_bounds=(0.0, np.inf),
        insolation_bounds=(0.0, np.inf),
    )


def filter_rocky_catalog(raw, curve_mass, curve_radius, *, shift=0.0, cuts=None):
    """Return the quality-controlled sample and its rocky subset."""
    cuts = DEFAULT_ROCKY_CATALOG_CUTS if cuts is None else cuts
    frame = raw.copy()
    if cuts["exclude_mass_limits"]:
        frame = frame[~frame["mass_limit_flag"].fillna(0).ne(0)].copy()
    if cuts["exclude_radius_limits"]:
        frame = frame[~frame["radius_limit_flag"].fillna(0).ne(0)].copy()
    if cuts["require_two_sided_mass"]:
        frame = frame[
            frame["mass_error_plus"].notna() & frame["mass_error_minus"].notna()
        ].copy()
    if cuts["require_two_sided_radius"]:
        frame = frame[
            frame["radius_error_plus"].notna() & frame["radius_error_minus"].notna()
        ].copy()

    mass_relative_error = frame["mass_error"] / frame["mass"].abs()
    frame = frame[
        mass_relative_error.isna()
        | mass_relative_error.le(cuts["max_mass_relative_uncertainty"])
    ].copy()
    radius_relative_error = frame["radius_error"] / frame["radius"].abs()
    frame = frame[
        radius_relative_error.isna()
        | radius_relative_error.le(cuts["max_radius_relative_uncertainty"])
    ].copy()
    frame = add_stellar_type(frame)
    frame["planet_label"] = frame.get(
        "planet_name", pd.Series("", index=frame.index)
    ).fillna("").astype(str)
    frame["discovery_facility"] = frame.get(
        "discovery_facility", pd.Series("Unknown", index=frame.index)
    ).fillna("Unknown").astype(str)
    threshold = radius_on_curve(frame["mass"], curve_mass, curve_radius, shift=shift)
    frame["rocky_threshold_radius"] = threshold
    frame["below_rocky_threshold"] = (
        frame["radius"].to_numpy() <= threshold
    ) & np.isfinite(threshold)
    output = frame.rename(columns={
        "host_name": "host_name", "mass": "mass_p", "radius": "radius_p",
        "insolation": "flux_p", "mass_error_plus": "mass_err_plus",
        "mass_error_minus": "mass_err_minus", "radius_error_plus": "radius_err_plus",
        "radius_error_minus": "radius_err_minus", "mass_provenance": "mass_provider",
        "effective_temperature": "teff_s", "stellar_radius": "radius_s",
        "stellar_mass": "mass_s", "mass_limit_flag": "mass_limit_flag",
        "radius_limit_flag": "radius_limit_flag",
        "insolation_limit_flag": "insolation_limit_flag",
        "mass_reference": "mass_reference", "radius_reference": "radius_reference",
    })
    return output, output[output["below_rocky_threshold"]].copy()


def restrict_science_window(
    frame,
    *,
    insolation=(0.1, 1e4),
    radius=(0.6, 2.2),
    stellar_types=("F", "G", "K", "M"),
):
    """Apply the declared radius, insolation, and stellar-type study window."""
    return frame[
        frame["flux_p"].between(*insolation, inclusive="neither")
        & frame["radius_p"].between(*radius, inclusive="neither")
        & frame["stype_clean"].isin(stellar_types)
    ].copy()


def load_and_filter_exoplanets(csv_path, instrument="LIFE") -> pd.DataFrame:
    """Return a NASA catalog normalized for a supported survey detector."""
    frame = read_nasa_csv(csv_path)
    instrument_key = instrument.upper()
    supported = set(FACILITY_BY_INSTRUMENT) | set(DIRECT_IMAGING_WAVELENGTH_M)
    if instrument_key not in supported:
        choices = ", ".join(sorted(supported))
        raise ValueError(f"Unknown instrument: {instrument}. Use one of {choices}.")

    facility = FACILITY_BY_INSTRUMENT.get(instrument_key)
    if facility is not None:
        if "disc_facility" not in frame.columns:
            raise KeyError(
                f"{csv_path} has no disc_facility column, so {instrument} "
                "discoveries cannot be selected from it."
            )
        frame = frame[frame["disc_facility"] == facility].copy()

    angular_rows = frame["sy_dist"].notnull() & frame["pl_orbsmax"].notnull()
    frame["AngSep"] = np.nan
    frame.loc[angular_rows, "AngSep"] = (
        frame.loc[angular_rows, "pl_orbsmax"]
        / frame.loc[angular_rows, "sy_dist"]
        * 206265
    )

    flux_rows = (
        frame["pl_rade"].notnull() & frame["st_rad"].notnull()
        & frame["pl_eqt"].notnull() & frame["st_teff"].notnull()
    )
    wavelength = DIRECT_IMAGING_WAVELENGTH_M.get(instrument_key)
    for column in ("Fp", "fp", "flux_ratio_value_best"):
        frame[column] = np.nan
    if wavelength is not None:
        planet_radiance = blackbody_spectral_radiance(
            wavelength, frame.loc[flux_rows, "pl_eqt"]
        )
        star_radiance = blackbody_spectral_radiance(
            wavelength, frame.loc[flux_rows, "st_teff"]
        )
        planet_radius_m = frame.loc[flux_rows, "pl_rade"] * const.R_earth
        star_radius_m = frame.loc[flux_rows, "st_rad"] * const.R_sun
        contrast = (planet_radius_m / star_radius_m) ** 2 * (
            planet_radiance / star_radiance
        )
        for column in ("Fp", "fp", "flux_ratio_value_best"):
            frame.loc[flux_rows, column] = contrast

    required = ["pl_rade", "sy_dist", "st_rad", "pl_eqt", "st_teff"]
    frame = frame[~frame[required].isnull().any(axis=1)].copy()
    column_map = {
        "luminosity_s": lambda data: (
            10 ** data["st_lum"] if "st_lum" in data.columns else np.nan
        ),
        "distance_s": "sy_dist", "radius_p": "pl_rade", "radius_s": "st_rad",
        "p_orb": "pl_orbper", "semimajor_p": "pl_orbsmax", "temp_s": "st_teff",
        "temp_p": "pl_eqt", "mass_p": "pl_bmasse", "detected": True,
        "detected_best": True, "detected_worst": True,
        "flux_ratio_value_best": "flux_ratio_value_best", "maxangsep": "AngSep",
        "z": 0.1, "stype": "G",
        "habitable": lambda data: (
            (data["pl_rade"] < 1.5) & data["pl_eqt"].between(270, 390)
        ),
        "run": 0, "Rp": "pl_rade", "Porb": "pl_orbper", "Mp": "pl_bmasse",
        "ep": 0.0, "ecc_p": 0.0, "ip": 0.0, "inc_p": 0.0,
        "Omegap": 0.0, "large_omega_p": 0.0, "omegap": 0.0,
        "small_omega_p": 0.0, "thetap": "AngSep", "Abond": 0.3,
        "AgeomVIS": 0.3, "AgeomMIR": 0.3, "ap": "pl_orbsmax",
        "rp": "pl_orbsmax", "AngSep": "AngSep", "maxAngSep": "AngSep",
        "Fp": "Fp", "fp": "fp", "Tp": "pl_eqt", "Msun": "st_mass",
        "Nuniverse": 1, "Nstar": 1, "nstar": 1, "ra": 0.0, "dec": 0.0,
        "name_s": "st_refname", "id": lambda data: np.arange(len(data)),
        "Rs": "st_rad", "Ms": "st_mass", "Ts": "st_teff", "Ds": "sy_dist",
        "RA": 0.0, "Dec": 0.0,
    }
    for column, source in column_map.items():
        if callable(source):
            frame[column] = source(frame)
        elif isinstance(source, str) and source in frame.columns:
            frame[column] = frame[source]
        else:
            frame[column] = source

    frame.loc[frame["luminosity_s"] < 0.1, "stype"] = "M"
    frame.loc[
        frame["luminosity_s"].between(0.1, 0.6, inclusive="left"), "stype"
    ] = "K"
    numeric_columns = [
        "radius_p", "Rp", "Porb", "Mp", "semimajor_p", "ap", "rp", "temp_p",
        "Tp", "temp_s", "Ts", "radius_s", "Rs", "Ms", "Msun", "distance_s",
        "Ds", "luminosity_s", "mass_p",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["radius_bin"] = pd.cut(
        frame["radius_p"], [0, 1.5, 3.0, 6.0],
        labels=["<1.5", "1.5–3.0", "3.0–6.0"], include_lowest=True,
    )
    return frame


def load_and_filter_nasa(csv_path, m_ref, r_ref, shift: float,
                         redownload: bool = False,
                         download_if_missing: bool = False,
                         exclude_mass_limits: bool = True,
                         exclude_radius_limits: bool = True,
                         require_two_sided_mass: bool = True,
                         require_two_sided_radius: bool = True,
                         max_mass_rel_uncertainty: float = 0.5,
                         max_radius_rel_uncertainty: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and filter NASA rocky planets catalog with quality cuts and threshold.

    Returns (all_quality_filtered, rocky_filtered) DataFrames filtered against
    a rocky mass-radius reference curve with optional vertical shift.
    """
    raw = load_nasa_rocky_source(
        csv_path,
        redownload=redownload,
        download_if_missing=download_if_missing,
    )
    cuts = {**DEFAULT_ROCKY_CATALOG_CUTS,
            "exclude_mass_limits": exclude_mass_limits,
            "exclude_radius_limits": exclude_radius_limits,
            "require_two_sided_mass": require_two_sided_mass,
            "require_two_sided_radius": require_two_sided_radius,
            "max_mass_relative_uncertainty": max_mass_rel_uncertainty,
            "max_radius_relative_uncertainty": max_radius_rel_uncertainty}
    return filter_rocky_catalog(raw, m_ref, r_ref, shift=shift, cuts=cuts)


__all__ = [
    "COMPARISON_PARAMETER_BOX", "DEFAULT_ROCKY_CATALOG_CUTS",
    "FGKM_TEMPERATURE_BOUNDS", "filter_rocky_catalog",
    "load_and_filter_exoplanets", "load_and_filter_nasa", "load_measured_planets",
    "load_nasa_rocky_source", "parameter_box_mask", "read_nasa_csv",
    "nasa_tap_url", "restrict_science_window",
]

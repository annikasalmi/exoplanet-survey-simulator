"""
50_ppop_nasa_dace_3x3.py

Purpose
-------
Download a safer NASA PSCompPars table with error columns AND limit flags,
cache it in:
    run/kepler/data/NASA/NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv

Then make the 3x3 mass-radius-insolation plots:
    Row 1 = P-Pop Kepler-detected planets
    Row 2 = NASA PSCompPars planets, with unsafe mass/radius rows removed
    Row 3 = DACE planets

Columns:
    I < 10, I < 50, I > 50

-----------------------
Some NASA PSCompPars masses are upper/lower limits. Example:
    Kepler-409 b: M_p < 6 M_Earth
    Kepler-106 d: M_p < 8.1 M_Earth

If we plot those as normal points at mass = 6 or 8.1, the planet can look
"below the rocky curve" by mistake. This script prevents that by downloading
and using:
    pl_bmasselim, pl_radelim, pl_insollim
plus the uncertainty columns:
    pl_bmasseerr1/2, pl_radeerr1/2

Caveman version:
    Old CSV lost NASA warning labels.
    New CSV keeps warning labels.
    Bad NASA mass? Do not draw it as a normal rock point.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================


def find_project_root(start_path: Path) -> Path:
    """Walk upward until we find the project root containing run/kepler."""
    start_path = start_path.resolve()
    for p in [start_path] + list(start_path.parents):
        if (p / "run" / "kepler").exists():
            return p
    return Path.cwd()


ROOT = find_project_root(Path(__file__).resolve())

PPOP_DATA_DIR = ROOT / "run" / "kepler" / "data" / "Gaia"
NASA_DATA_DIR = ROOT / "run" / "kepler" / "data" / "NASA"
DACE_DATA_DIR = ROOT / "run" / "kepler" / "data" / "DACE"
REF_CURVE_PATH = ROOT / "run" / "kepler" / "reference_curves" / "ref.ddat"

NASA_DATA_DIR.mkdir(parents=True, exist_ok=True)
DACE_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUT_DIR = ROOT / "output/plots" / "50_ppop_nasa_dace_3x3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# New safer NASA cache. This is the file this script downloads and then uses.
NASA_PSCOMPPARS_FLAGS_CACHE = (
    NASA_DATA_DIR / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv"
)

FORCE_REDOWNLOAD_NASA = False
DOWNLOAD_NASA_IF_MISSING = True

# NASA plot versions. The attached older script made ALL and KEPLERONLY plots;
# this version keeps that behavior.
NASA_VERSIONS = [
    (None, "ALL", "NASA PSCompPars strict: all facilities"),
    ("Kepler", "KEPLERONLY", "NASA PSCompPars strict: Kepler only"),
]

# Strict NASA mass-radius quality rules.
# These are for rocky-curve / composition plots, where false x-position is dangerous.
NASA_REQUIRE_STRICT_MR_QUALITY = True
NASA_EXCLUDE_MASS_LIMITS = True
NASA_EXCLUDE_RADIUS_LIMITS = True
NASA_EXCLUDE_INSOLATION_LIMITS = False  # Usually keep; insolation bins tolerate this better than M-R claims.
NASA_EXCLUDE_CALCULATED_MASSES = True
NASA_REQUIRE_TWO_SIDED_MASS_ERRORS = True
NASA_REQUIRE_TWO_SIDED_RADIUS_ERRORS = True

# Uncertainty cuts, kept from your attached script.
MAX_RADIUS_REL_UNCERTAINTY = 0.08  # 8%
MAX_MASS_REL_UNCERTAINTY = 0.25    # 25%

# P-Pop settings.
PPOP_DETECTION_COLUMNS = ["detected", "detected_best"]
PPOP_FALLBACK_PATHS = [
    ROOT / "output/plots" / "19_w3_nasa_vs_ppop_notebook_style" / "plot_data_ppop_detected.csv",
    ROOT / "output/plots" / "21_ppop_nasa_dace_insolation_3x3" / "plot_data_ppop_detected.csv",
    ROOT / "output/plots" / "w2_nasa_real_vs_ppop_insolation" / "plot_data_ppop_detected.csv",
    ROOT / "output/plots" / "w2_nasa_pscomppars_vs_ppop_insolation" / "plot_data_ppop_detected.csv",
]

# DACE settings. This follows the same spirit as your attached script.
DACE_SOURCE_MODE = "query_or_cache"  # query_or_cache, cache, or query
DACE_NOTEBOOK_RAW_CACHE = DACE_DATA_DIR / "DACE_planets_raw_from_notebook.csv"
DACE_CANDIDATE_PATHS = [
    DACE_NOTEBOOK_RAW_CACHE,
    DACE_DATA_DIR / "DACE_planets_raw_from_notebook(1).csv",
    DACE_DATA_DIR / "dace_sample.csv",
    DACE_DATA_DIR / "dace_sample(1).csv",
    ROOT / "dace_sample.csv",
    Path(__file__).resolve().parent / "dace_sample.csv",
]
DACE_QUERY_MASS_LIMIT_MJUP = 0.035
MJUP_TO_MEARTH = 317.828
RJUP_TO_REARTH = 11.209

# Plot camera.
NOTEBOOK_XLIM = (0.0, 12.0)
NOTEBOOK_YLIM = (0.5, 2.2)

# Plot style.
CMAP = "plasma"
POINT_SIZE_PPOP = 22
POINT_SIZE_NASA = 24
POINT_SIZE_DACE = 28
POINT_ALPHA = 0.78
COLOR_PERCENTILES = (1, 99)
DRAW_ROCKY_CURVE = True
ALLOW_TOY_ROCKY_CURVE_IF_REF_MISSING = True

DRAW_ERRORBARS = True
ERRORBAR_COLOR = "0.40"
ERRORBAR_ALPHA = 0.35
ERRORBAR_LINEWIDTH = 0.45
ERRORBAR_CAPSIZE = 0

LABEL_BELOW_CURVE_FOR = {"NASA PSCompPars strict: all facilities", "NASA PSCompPars strict: Kepler only", "DACE notebook query", "DACE sample"}
BELOW_CURVE_TOLERANCE_REARTH = 0.0
LABEL_FONT_SIZE = 4.5
LABEL_DX = 0.05
LABEL_DY = 0.025

# Constants for fallback insolation calculations.
T_SUN_K = 5772.0
EARTH_EQ_TEMP_K = 278.5


# ============================================================
# Basic helpers
# ============================================================


def first_existing_path(paths, label: str) -> Path:
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Could not find {label}. Tried:\n" + "\n".join(f"  - {Path(p)}" for p in paths)
    )


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if s.dtype == object:
        ss = s.astype(str).str.strip().str.lower()
        return ss.isin(["true", "1", "yes", "y", "t"])
    return s.fillna(0).astype(bool)


def force_numeric(df: pd.DataFrame, columns) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def first_column(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def first_available_series(df: pd.DataFrame, candidates, default=np.nan):
    for c in candidates:
        if c in df.columns:
            return df[c]
    return pd.Series(default, index=df.index)


def require_columns(df: pd.DataFrame, cols, label: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}\n\nAvailable columns:\n{df.columns.tolist()}"
        )


# ============================================================
# NASA PSCompPars download with error columns + limit flags
# ============================================================


def build_nasa_pscomppars_query() -> str:
    """
    NASA table query.

    Caveman:
        Get planet numbers.
        Get error bars.
        Get warning labels for limits.
        No warning label lost.
    """
    return """
    SELECT
        pl_name,
        hostname,
        discoverymethod,
        disc_facility,
        disc_telescope,
        tran_flag,

        pl_orbper,
        pl_orbsmax,
        pl_orbincl,
        pl_trandep,
        pl_trandur,

        pl_insol,
        pl_insolerr1,
        pl_insolerr2,
        pl_insollim,
        pl_eqt,

        pl_rade,
        pl_radeerr1,
        pl_radeerr2,
        pl_radelim,
        pl_rade_reflink,

        pl_bmasse,
        pl_bmasseerr1,
        pl_bmasseerr2,
        pl_bmasselim,
        pl_bmassprov,
        pl_bmasse_reflink,

        st_rad,
        st_mass,
        st_teff,
        st_lum,
        sy_dist,
        sy_kepmag,
        sy_gaiamag
    FROM pscomppars
    WHERE tran_flag = 1
      AND pl_rade IS NOT NULL
      AND pl_bmasse IS NOT NULL
      AND pl_insol IS NOT NULL
    """


def nasa_tap_url(query: str) -> str:
    return "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=" + quote(query) + "&format=csv"


def download_nasa_pscomppars_with_flags(cache_path: Path = NASA_PSCOMPPARS_FLAGS_CACHE) -> pd.DataFrame:
    query = build_nasa_pscomppars_query()
    url = nasa_tap_url(query)

    print("Downloading NASA PSCompPars with error columns + limit flags...")
    print("Saving to:", cache_path)
    df = pd.read_csv(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Downloaded NASA rows: {len(df):,}")
    return df


def load_nasa_pscomppars_with_flags() -> pd.DataFrame:
    if FORCE_REDOWNLOAD_NASA or not NASA_PSCOMPPARS_FLAGS_CACHE.exists():
        if not DOWNLOAD_NASA_IF_MISSING and not NASA_PSCOMPPARS_FLAGS_CACHE.exists():
            raise FileNotFoundError(
                f"NASA cache missing and DOWNLOAD_NASA_IF_MISSING=False:\n{NASA_PSCOMPPARS_FLAGS_CACHE}"
            )
        return download_nasa_pscomppars_with_flags(NASA_PSCOMPPARS_FLAGS_CACHE)

    print("Loading cached NASA PSCompPars with flags:")
    print(NASA_PSCOMPPARS_FLAGS_CACHE)
    return pd.read_csv(NASA_PSCOMPPARS_FLAGS_CACHE)


# ============================================================
# Uncertainty/error-bar helpers
# ============================================================


def _numeric_or_nan(df: pd.DataFrame, col: str | None, index=None) -> pd.Series:
    if index is None:
        index = df.index
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _first_existing_col(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def choose_abs_error_pair(
    df: pd.DataFrame,
    value_col: str,
    lower_candidates,
    upper_candidates,
    symmetric_candidates=(),
    rel_candidates=(),
    unit_multiplier: float = 1.0,
):
    idx = df.index
    value = pd.to_numeric(df[value_col], errors="coerce")

    lower_col = _first_existing_col(df, lower_candidates)
    upper_col = _first_existing_col(df, upper_candidates)
    sym_col = _first_existing_col(df, symmetric_candidates)
    rel_col = _first_existing_col(df, rel_candidates)

    err_minus = _numeric_or_nan(df, lower_col, idx).abs() * unit_multiplier
    err_plus = _numeric_or_nan(df, upper_col, idx).abs() * unit_multiplier

    if sym_col is not None:
        sym = _numeric_or_nan(df, sym_col, idx).abs() * unit_multiplier
        err_minus = err_minus.fillna(sym)
        err_plus = err_plus.fillna(sym)

    if rel_col is not None:
        rel = _numeric_or_nan(df, rel_col, idx).abs()
        rel = rel.where(rel <= 1.0, rel / 100.0)
        rel_abs = rel * value.abs()
        err_minus = err_minus.fillna(rel_abs)
        err_plus = err_plus.fillna(rel_abs)

    return err_minus, err_plus


def add_uncertainty_columns(
    df: pd.DataFrame,
    dataset_label: str,
    mass_lower_candidates=(),
    mass_upper_candidates=(),
    mass_symmetric_candidates=(),
    mass_rel_candidates=(),
    radius_lower_candidates=(),
    radius_upper_candidates=(),
    radius_symmetric_candidates=(),
    radius_rel_candidates=(),
    mass_error_unit_multiplier: float = 1.0,
    radius_error_unit_multiplier: float = 1.0,
) -> pd.DataFrame:
    df = df.copy()
    if "mass_p" not in df.columns or "radius_p" not in df.columns:
        return df

    df["mass_err_minus"], df["mass_err_plus"] = choose_abs_error_pair(
        df,
        value_col="mass_p",
        lower_candidates=mass_lower_candidates,
        upper_candidates=mass_upper_candidates,
        symmetric_candidates=mass_symmetric_candidates,
        rel_candidates=mass_rel_candidates,
        unit_multiplier=mass_error_unit_multiplier,
    )
    df["radius_err_minus"], df["radius_err_plus"] = choose_abs_error_pair(
        df,
        value_col="radius_p",
        lower_candidates=radius_lower_candidates,
        upper_candidates=radius_upper_candidates,
        symmetric_candidates=radius_symmetric_candidates,
        rel_candidates=radius_rel_candidates,
        unit_multiplier=radius_error_unit_multiplier,
    )

    mass_value = pd.to_numeric(df["mass_p"], errors="coerce").abs()
    radius_value = pd.to_numeric(df["radius_p"], errors="coerce").abs()

    mass_err_max = pd.concat([df["mass_err_minus"], df["mass_err_plus"]], axis=1).max(axis=1, skipna=True)
    radius_err_max = pd.concat([df["radius_err_minus"], df["radius_err_plus"]], axis=1).max(axis=1, skipna=True)

    df["mass_rel_uncertainty"] = mass_err_max / mass_value
    df["radius_rel_uncertainty"] = radius_err_max / radius_value
    df.loc[mass_err_max.isna(), "mass_rel_uncertainty"] = np.nan
    df.loc[radius_err_max.isna(), "radius_rel_uncertainty"] = np.nan

    df["has_mass_errorbar"] = df["mass_err_minus"].notna() | df["mass_err_plus"].notna()
    df["has_radius_errorbar"] = df["radius_err_minus"].notna() | df["radius_err_plus"].notna()
    df["has_two_sided_mass_errorbar"] = df["mass_err_minus"].notna() & df["mass_err_plus"].notna()
    df["has_two_sided_radius_errorbar"] = df["radius_err_minus"].notna() & df["radius_err_plus"].notna()
    df["has_any_errorbar"] = df["has_mass_errorbar"] | df["has_radius_errorbar"]
    df["uncertainty_dataset_label"] = dataset_label
    return df


def apply_uncertainty_filter(df: pd.DataFrame, dataset_label: str):
    df = df.copy()
    before = len(df)

    mass_rel = df.get("mass_rel_uncertainty", pd.Series(np.nan, index=df.index))
    radius_rel = df.get("radius_rel_uncertainty", pd.Series(np.nan, index=df.index))

    bad_mass = mass_rel.notna() & (mass_rel > MAX_MASS_REL_UNCERTAINTY)
    bad_radius = radius_rel.notna() & (radius_rel > MAX_RADIUS_REL_UNCERTAINTY)
    bad = bad_mass | bad_radius

    removed = df[bad].copy()
    kept = df[~bad].copy()

    kept["uncertainty_filter_status"] = "kept"
    removed["uncertainty_filter_status"] = "removed_high_uncertainty"

    print(f"{dataset_label} uncertainty filter: {before:,} -> {len(kept):,} kept; {len(removed):,} removed")
    print(f"  removed for mass uncertainty > {MAX_MASS_REL_UNCERTAINTY:.0%}: {int(bad_mass.sum()):,}")
    print(f"  removed for radius uncertainty > {MAX_RADIUS_REL_UNCERTAINTY:.0%}: {int(bad_radius.sum()):,}")
    return kept, removed


def errorbar_arrays(df: pd.DataFrame):
    if len(df) == 0:
        return None, None

    xminus = df.get("mass_err_minus", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    xplus = df.get("mass_err_plus", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    yminus = df.get("radius_err_minus", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    yplus = df.get("radius_err_plus", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)

    xerr = None
    yerr = None

    if np.isfinite(xminus).any() or np.isfinite(xplus).any():
        xerr = np.vstack([np.nan_to_num(xminus, nan=0.0), np.nan_to_num(xplus, nan=0.0)])
    if np.isfinite(yminus).any() or np.isfinite(yplus).any():
        yerr = np.vstack([np.nan_to_num(yminus, nan=0.0), np.nan_to_num(yplus, nan=0.0)])

    return xerr, yerr


# ============================================================
# NASA quality flags and loader
# ============================================================


def as_limit_flag(series: pd.Series) -> pd.Series:
    """NASA limit convention: 0 or NaN means not flagged as limit; nonzero means limit."""
    return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)


def add_nasa_quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["mass_is_limit"] = as_limit_flag(df.get("mass_limit_flag", pd.Series(0, index=df.index)))
    df["radius_is_limit"] = as_limit_flag(df.get("radius_limit_flag", pd.Series(0, index=df.index)))
    df["insolation_is_limit"] = as_limit_flag(df.get("insolation_limit_flag", pd.Series(0, index=df.index)))

    mass_ref = df.get("mass_reference", pd.Series("", index=df.index)).astype(str)
    radius_ref = df.get("radius_reference", pd.Series("", index=df.index)).astype(str)
    mass_provider = df.get("mass_provider", pd.Series("", index=df.index)).astype(str)

    df["mass_is_calculated"] = mass_ref.str.contains("CALCULATED_VALUE|Calculated Value", case=False, na=False)
    df["radius_is_calculated"] = radius_ref.str.contains("CALCULATED_VALUE|Calculated Value", case=False, na=False)

    # Provider is diagnostic, not a strict removal by itself.
    df["mass_provider_text"] = mass_provider

    df["mass_quality_reason"] = "good"
    df.loc[df["mass_is_limit"], "mass_quality_reason"] = "mass_limit_flag"
    df.loc[df["mass_is_calculated"], "mass_quality_reason"] = "calculated_mass_reference"
    if NASA_REQUIRE_TWO_SIDED_MASS_ERRORS:
        no_two_sided_mass = ~df.get("has_two_sided_mass_errorbar", pd.Series(False, index=df.index))
        df.loc[no_two_sided_mass & (df["mass_quality_reason"] == "good"), "mass_quality_reason"] = "missing_two_sided_mass_error"

    df["radius_quality_reason"] = "good"
    df.loc[df["radius_is_limit"], "radius_quality_reason"] = "radius_limit_flag"
    df.loc[df["radius_is_calculated"], "radius_quality_reason"] = "calculated_radius_reference"
    if NASA_REQUIRE_TWO_SIDED_RADIUS_ERRORS:
        no_two_sided_radius = ~df.get("has_two_sided_radius_errorbar", pd.Series(False, index=df.index))
        df.loc[no_two_sided_radius & (df["radius_quality_reason"] == "good"), "radius_quality_reason"] = "missing_two_sided_radius_error"

    mass_good = pd.Series(True, index=df.index)
    radius_good = pd.Series(True, index=df.index)
    insol_good = pd.Series(True, index=df.index)

    if NASA_EXCLUDE_MASS_LIMITS:
        mass_good &= ~df["mass_is_limit"]
    if NASA_EXCLUDE_CALCULATED_MASSES:
        mass_good &= ~df["mass_is_calculated"]
    if NASA_REQUIRE_TWO_SIDED_MASS_ERRORS:
        mass_good &= df.get("has_two_sided_mass_errorbar", pd.Series(False, index=df.index))

    if NASA_EXCLUDE_RADIUS_LIMITS:
        radius_good &= ~df["radius_is_limit"]
    if NASA_REQUIRE_TWO_SIDED_RADIUS_ERRORS:
        radius_good &= df.get("has_two_sided_radius_errorbar", pd.Series(False, index=df.index))

    if NASA_EXCLUDE_INSOLATION_LIMITS:
        insol_good &= ~df["insolation_is_limit"]

    df["nasa_mass_good_for_rocky_curve"] = mass_good
    df["nasa_radius_good_for_rocky_curve"] = radius_good
    df["nasa_insolation_good_for_bins"] = insol_good
    df["nasa_good_for_mr_rocky_curve"] = mass_good & radius_good & insol_good

    return df


def standardize_nasa_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "pl_name": "planet_name",
        "hostname": "host_name",
        "discoverymethod": "discovery_method",
        "disc_facility": "discovery_facility",
        "pl_bmasse": "mass_p",
        "pl_bmasseerr1": "mass_err_plus",
        "pl_bmasseerr2": "mass_err_minus",
        "pl_bmasselim": "mass_limit_flag",
        "pl_bmassprov": "mass_provider",
        "pl_bmasse_reflink": "mass_reference",
        "pl_rade": "radius_p",
        "pl_radeerr1": "radius_err_plus",
        "pl_radeerr2": "radius_err_minus",
        "pl_radelim": "radius_limit_flag",
        "pl_rade_reflink": "radius_reference",
        "pl_insol": "flux_p",
        "pl_insolerr1": "flux_err_plus",
        "pl_insolerr2": "flux_err_minus",
        "pl_insollim": "insolation_limit_flag",
        "pl_orbper": "p_orb",
        "pl_orbsmax": "semimajor_p",
        "pl_orbincl": "inc_p",
        "st_rad": "radius_s",
        "st_mass": "mass_s",
        "st_teff": "teff_s",
        "st_lum": "st_lum_log10",
        "sy_dist": "distance_s",
        "sy_kepmag": "kepmag",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def load_nasa_observed(facility_filter: str | None, version_tag: str) -> pd.DataFrame:
    raw = load_nasa_pscomppars_with_flags()
    df = standardize_nasa_columns(raw)

    required_for_flags = ["mass_limit_flag", "radius_limit_flag", "insolation_limit_flag"]
    missing_flags = [c for c in required_for_flags if c not in df.columns]
    if missing_flags:
        raise ValueError(
            f"NASA table is missing limit flags {missing_flags}. Delete old cache and rerun with FORCE_REDOWNLOAD_NASA=True."
        )

    require_columns(df, ["mass_p", "radius_p", "flux_p"], "NASA PSCompPars")
    df = force_numeric(
        df,
        [
            "mass_p", "radius_p", "flux_p", "mass_err_plus", "mass_err_minus",
            "radius_err_plus", "radius_err_minus", "mass_limit_flag", "radius_limit_flag",
            "insolation_limit_flag", "p_orb", "semimajor_p", "radius_s", "mass_s",
            "teff_s", "distance_s", "kepmag",
        ],
    )

    before = len(df)
    df = df.dropna(subset=["mass_p", "radius_p", "flux_p"]).copy()
    df = df[(df["mass_p"] > 0) & (df["radius_p"] > 0) & (df["flux_p"] > 0)].copy()
    print(f"NASA valid M/R/insolation rows: {before:,} -> {len(df):,}")

    if facility_filter is not None:
        before = len(df)
        df = df[df["discovery_facility"].astype(str) == facility_filter].copy()
        print(f"NASA after discovery_facility == {facility_filter!r}: {before:,} -> {len(df):,}")

    df["planet_label"] = first_available_series(df, ["planet_name", "pl_name"], default="")
    df["comparison_group"] = "NASA PSCompPars"
    df["nasa_version_tag"] = version_tag

    # Error columns already renamed, but this normalizes signs and creates relative uncertainty.
    df = add_uncertainty_columns(
        df,
        "NASA PSCompPars",
        mass_lower_candidates=["mass_err_minus"],
        mass_upper_candidates=["mass_err_plus"],
        radius_lower_candidates=["radius_err_minus"],
        radius_upper_candidates=["radius_err_plus"],
    )

    df = add_nasa_quality_flags(df)

    # Save quality audit before removing anything.
    audit_path = OUT_DIR / f"nasa_{version_tag}_quality_audit_before_strict_filter.csv"
    df.to_csv(audit_path, index=False)
    print(f"Saved NASA quality audit before strict filter: {audit_path}")

    if NASA_REQUIRE_STRICT_MR_QUALITY:
        before = len(df)
        bad_quality = ~df["nasa_good_for_mr_rocky_curve"]
        removed_quality = df[bad_quality].copy()
        df = df[~bad_quality].copy()
        removed_quality_path = OUT_DIR / f"removed_nasa_{version_tag}_bad_mass_radius_quality.csv"
        removed_quality.to_csv(removed_quality_path, index=False)
        print(f"NASA strict M-R quality filter: {before:,} -> {len(df):,} kept; {len(removed_quality):,} removed")
        print("Removed NASA quality reasons:")
        reason_counts = (
            removed_quality[["mass_quality_reason", "radius_quality_reason"]]
            .value_counts(dropna=False)
            .reset_index(name="count")
        )
        print(reason_counts.to_string(index=False))
        print(f"Saved removed NASA bad-quality rows: {removed_quality_path}")

    df, removed_unc = apply_uncertainty_filter(df, f"NASA {version_tag}")
    removed_unc.to_csv(OUT_DIR / f"removed_nasa_{version_tag}_high_uncertainty.csv", index=False)

    print(f"NASA {version_tag} rows kept for plotting: {len(df):,}")
    return df


# ============================================================
# P-Pop loader
# ============================================================


def load_ppop_detected() -> pd.DataFrame:
    pattern = str(PPOP_DATA_DIR / "kepler_catalog_*.csv")
    files = sorted(glob.glob(pattern))

    if files:
        dfs = []
        for f in files:
            d = pd.read_csv(f)
            d["source_file"] = Path(f).name
            dfs.append(d)
        df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(files)} P-Pop Kepler catalog file(s). Rows before detected filter: {len(df):,}")
    else:
        fallback = first_existing_path(PPOP_FALLBACK_PATHS, "P-Pop detected plot data")
        df = pd.read_csv(fallback)
        print(f"No raw P-Pop catalogs found. Loaded P-Pop fallback: {fallback}")

    if "flux_p" not in df.columns:
        if "pl_insol" in df.columns:
            df = df.rename(columns={"pl_insol": "flux_p"})
        elif "insolation" in df.columns:
            df = df.rename(columns={"insolation": "flux_p"})

    detected_col = None
    for c in PPOP_DETECTION_COLUMNS:
        if c in df.columns:
            detected_col = c
            break

    if detected_col is not None:
        df = df[to_bool_series(df[detected_col])].copy()
        print(f"P-Pop detected column used: {detected_col}")
    else:
        print("P-Pop fallback appears already detected-filtered; no detected column used.")

    require_columns(df, ["mass_p", "radius_p", "flux_p"], "P-Pop")
    df = force_numeric(df, ["mass_p", "radius_p", "flux_p"])
    df = df.dropna(subset=["mass_p", "radius_p", "flux_p"]).copy()
    df = df[(df["mass_p"] > 0) & (df["radius_p"] > 0) & (df["flux_p"] > 0)].copy()

    df["planet_label"] = first_available_series(df, ["planet_name", "pl_name", "name", "id"], default="")
    df["comparison_group"] = "P-Pop detected"
    df = add_uncertainty_columns(df, "P-Pop")

    # Missing uncertainty is okay for simulated P-Pop.
    df, removed = apply_uncertainty_filter(df, "P-Pop")
    removed.to_csv(OUT_DIR / "removed_ppop_high_uncertainty.csv", index=False)

    print(f"P-Pop rows kept: {len(df):,}")
    return df


# ============================================================
# DACE loader
# ============================================================


def estimate_semimajor_from_period(period_days, stellar_mass_solar=None):
    p_days = pd.to_numeric(period_days, errors="coerce")
    p_year = p_days / 365.25
    if stellar_mass_solar is None:
        mstar = pd.Series(1.0, index=p_days.index)
    else:
        mstar = pd.to_numeric(stellar_mass_solar, errors="coerce").fillna(1.0)
    a = (mstar * p_year ** 2) ** (1.0 / 3.0)
    return a.replace([np.inf, -np.inf], np.nan)


def calc_insolation_from_star_and_orbit(stellar_radius, stellar_teff, semi_major_axis):
    rstar = pd.to_numeric(stellar_radius, errors="coerce")
    teff = pd.to_numeric(stellar_teff, errors="coerce")
    a = pd.to_numeric(semi_major_axis, errors="coerce")
    flux = (rstar ** 2) * (teff / T_SUN_K) ** 4 / (a ** 2)
    flux = flux.replace([np.inf, -np.inf], np.nan)
    flux[flux <= 0] = np.nan
    return flux


def calc_insolation_from_teq(equilibrium_temp):
    teq = pd.to_numeric(equilibrium_temp, errors="coerce")
    flux = (teq / EARTH_EQ_TEMP_K) ** 4
    flux = flux.replace([np.inf, -np.inf], np.nan)
    flux[flux <= 0] = np.nan
    return flux


def choose_flux_source(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flux = pd.Series(np.nan, index=df.index, dtype=float)
    source = pd.Series("missing", index=df.index, dtype=object)

    for c in ["flux_p", "pl_insol", "insolation_flux", "insolation_flux_computed", "insolation"]:
        if c in df.columns:
            values = pd.to_numeric(df[c], errors="coerce")
            use = flux.isna() & values.notna() & (values > 0)
            flux[use] = values[use]
            source[use] = f"catalog_{c}"

    rstar_col = first_column(df, ["radius_s", "st_rad", "stellar_radius"])
    teff_col = first_column(df, ["teff_s", "temp_s", "st_teff", "stellar_eff_temp"])
    a_col = first_column(df, ["semimajor_p", "pl_orbsmax", "semi_major_axis"])
    p_col = first_column(df, ["p_orb", "pl_orbper", "period"])
    mstar_col = first_column(df, ["mass_s", "st_mass", "stellar_mass"])
    teq_col = first_column(df, ["temp_p", "pl_eqt", "equilibrium_temp"])

    if rstar_col and teff_col and a_col:
        flux_calc = calc_insolation_from_star_and_orbit(df[rstar_col], df[teff_col], df[a_col])
        use = flux.isna() & flux_calc.notna()
        flux[use] = flux_calc[use]
        source[use] = "computed_from_Rstar_Teff_a"

    if rstar_col and teff_col and p_col:
        if mstar_col:
            a_est = estimate_semimajor_from_period(df[p_col], df[mstar_col])
            a_source = "estimated_a_from_period_and_Mstar"
        else:
            a_est = estimate_semimajor_from_period(df[p_col], None)
            a_source = "estimated_a_from_period_assuming_1Msun"
        flux_calc = calc_insolation_from_star_and_orbit(df[rstar_col], df[teff_col], a_est)
        use = flux.isna() & flux_calc.notna()
        flux[use] = flux_calc[use]
        source[use] = a_source

    if teq_col:
        flux_teq = calc_insolation_from_teq(df[teq_col])
        use = flux.isna() & flux_teq.notna()
        flux[use] = flux_teq[use]
        source[use] = "estimated_from_equilibrium_temp"

    df["flux_p"] = flux
    df["insolation_source"] = source
    return df


def query_dace_like_notebook() -> pd.DataFrame:
    try:
        from dace_query.exoplanet import Exoplanet
    except ImportError as exc:
        raise ImportError(
            "Could not import dace_query. Use DACE_SOURCE_MODE='query_or_cache' with a DACE cache/sample file."
        ) from exc

    print("Querying DACE with notebook technique...")
    df = Exoplanet.query_database(output_format="pandas")
    DACE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DACE_NOTEBOOK_RAW_CACHE, index=False)
    print(f"Saved raw DACE query cache: {DACE_NOTEBOOK_RAW_CACHE}")
    return df


def load_dace_cache_or_sample() -> tuple[pd.DataFrame, str]:
    path = first_existing_path(DACE_CANDIDATE_PATHS, "DACE raw notebook CSV/cache or DACE sample")
    print(f"Loading DACE cache/sample: {path}")
    return pd.read_csv(path), str(path)


def clean_dace_after_standardization(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(df, ["mass_p", "radius_p", "flux_p"], "DACE")
    df = force_numeric(df, ["mass_p", "radius_p", "flux_p"])
    before_clean = len(df)
    df = df.dropna(subset=["mass_p", "radius_p", "flux_p"]).copy()
    df = df[(df["mass_p"] > 0) & (df["radius_p"] > 0) & (df["flux_p"] > 0)].copy()
    print(f"DACE rows after valid M/R/insolation: {before_clean:,} -> {len(df):,}")

    df, removed = apply_uncertainty_filter(df, "DACE")
    removed.to_csv(OUT_DIR / "removed_dace_high_uncertainty.csv", index=False)
    if "insolation_source" in df.columns:
        print("DACE insolation source counts:")
        print(df["insolation_source"].value_counts(dropna=False).to_string())
    return df


def standardize_dace_notebook_raw(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required_raw = ["planet_mass", "planet_radius", "insolation_flux"]
    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        raise ValueError(f"Raw DACE notebook/query dataframe is missing: {missing}")

    df["planet_mass"] = pd.to_numeric(df["planet_mass"], errors="coerce")
    before = len(df)
    df = df[df["planet_mass"] < DACE_QUERY_MASS_LIMIT_MJUP].copy()
    print(f"DACE mass filter planet_mass < {DACE_QUERY_MASS_LIMIT_MJUP} M_Jup: {before:,} -> {len(df):,}")

    df["mass_p"] = pd.to_numeric(df["planet_mass"], errors="coerce") * MJUP_TO_MEARTH
    df["radius_p"] = pd.to_numeric(df["planet_radius"], errors="coerce") * RJUP_TO_REARTH
    df["flux_p"] = pd.to_numeric(df["insolation_flux"], errors="coerce")
    df["planet_label"] = first_available_series(df, ["planet_name", "pl_name"], default="")
    df["insolation_source"] = "catalog_insolation_flux_notebook"
    df["comparison_group"] = "DACE notebook query"

    df = add_uncertainty_columns(
        df,
        "DACE notebook raw",
        mass_lower_candidates=["planet_mass_lower"],
        mass_upper_candidates=["planet_mass_upper"],
        mass_symmetric_candidates=["planet_mass_error"],
        mass_rel_candidates=["planet_mass_rel_err"],
        radius_lower_candidates=["planet_radius_lower"],
        radius_upper_candidates=["planet_radius_upper"],
        radius_symmetric_candidates=["planet_radius_error"],
        radius_rel_candidates=["planet_radius_rel_err"],
        mass_error_unit_multiplier=MJUP_TO_MEARTH,
        radius_error_unit_multiplier=RJUP_TO_REARTH,
    )
    return clean_dace_after_standardization(df)


def standardize_dace_sample_earth_units(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mass_p"] = first_available_series(df, ["mass_p", "planet_mass", "pl_bmasse"])
    df["radius_p"] = first_available_series(df, ["radius_p", "planet_radius", "pl_rade"])
    df["planet_label"] = first_available_series(df, ["planet_name", "pl_name"], default="")
    df = choose_flux_source(df)
    df["comparison_group"] = "DACE sample"

    df = force_numeric(df, ["mass_p", "radius_p", "flux_p"])
    df = add_uncertainty_columns(
        df,
        "DACE sample",
        mass_lower_candidates=["planet_mass_lower", "mass_err_minus", "pl_bmasseerr2"],
        mass_upper_candidates=["planet_mass_upper", "mass_err_plus", "pl_bmasseerr1"],
        mass_symmetric_candidates=["planet_mass_error", "mass_error"],
        mass_rel_candidates=["planet_mass_rel_err", "mass_rel_uncertainty"],
        radius_lower_candidates=["planet_radius_lower", "radius_err_minus", "pl_radeerr2"],
        radius_upper_candidates=["planet_radius_upper", "radius_err_plus", "pl_radeerr1"],
        radius_symmetric_candidates=["planet_radius_error", "radius_error"],
        radius_rel_candidates=["planet_radius_rel_err", "radius_rel_uncertainty"],
    )
    return clean_dace_after_standardization(df)


def load_dace_sample() -> pd.DataFrame:
    if DACE_SOURCE_MODE not in {"query", "query_or_cache", "cache"}:
        raise ValueError("DACE_SOURCE_MODE must be 'query', 'query_or_cache', or 'cache'.")

    if DACE_SOURCE_MODE == "cache":
        raw, raw_origin = load_dace_cache_or_sample()
    else:
        try:
            raw = query_dace_like_notebook()
            raw_origin = "live_query"
        except Exception as exc:
            if DACE_SOURCE_MODE == "query":
                raise
            print("WARNING: live DACE query failed; falling back to cache/sample.")
            print(f"Reason: {type(exc).__name__}: {exc}")
            raw, raw_origin = load_dace_cache_or_sample()

    origin_lower = str(raw_origin).lower()
    raw_cols = set(raw.columns)
    has_raw_notebook_columns = {"planet_mass", "planet_radius", "insolation_flux"}.issubset(raw_cols)
    is_raw_notebook_cache = "dace_planets_raw_from_notebook" in origin_lower

    if (raw_origin == "live_query" or is_raw_notebook_cache) and has_raw_notebook_columns:
        print("DACE input = raw notebook/Jupiter-unit table.")
        return standardize_dace_notebook_raw(raw)

    print("DACE input = cleaned/sample Earth-unit table.")
    return standardize_dace_sample_earth_units(raw)


# ============================================================
# Rocky curve and diagnostics
# ============================================================


def load_rocky_reference_curve(ref_path=REF_CURVE_PATH):
    ref_path = Path(ref_path)
    if not ref_path.exists():
        if not ALLOW_TOY_ROCKY_CURVE_IF_REF_MISSING:
            raise FileNotFoundError(f"Could not find rocky reference table: {ref_path}")
        print("WARNING: ref.ddat not found. Using toy rocky curve fallback for testing.")
        m_ref = np.linspace(0.05, 100.0, 2000)
        r_ref = m_ref ** 0.27
        return m_ref, r_ref

    ref = np.loadtxt(ref_path, comments="#")
    if ref.ndim == 1:
        ref = ref.reshape(1, -1)
    if ref.shape[1] < 2:
        raise ValueError("ref.ddat must have at least two columns: mass and radius.")

    m_ref = ref[:, 0].astype(float)
    r_ref = ref[:, 1].astype(float)
    good = np.isfinite(m_ref) & np.isfinite(r_ref) & (m_ref > 0) & (r_ref > 0)
    m_ref, r_ref = m_ref[good], r_ref[good]
    order = np.argsort(m_ref)
    return m_ref[order], r_ref[order]


def rocky_radius_from_mass(m_planet, m_ref=None, r_ref=None):
    if m_ref is None or r_ref is None:
        m_ref, r_ref = load_rocky_reference_curve()
    return np.interp(np.asarray(m_planet, dtype=float), m_ref, r_ref, left=np.nan, right=np.nan)


def add_rocky_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    m_ref, r_ref = load_rocky_reference_curve()
    df["rocky_radius_ref"] = rocky_radius_from_mass(df["mass_p"], m_ref, r_ref)
    df["radius_excess_to_rocky"] = df["radius_p"] - df["rocky_radius_ref"]
    df["radius_ratio_to_rocky"] = df["radius_p"] / df["rocky_radius_ref"]
    df["below_rocky_curve"] = df["radius_p"] < (df["rocky_radius_ref"] - BELOW_CURVE_TOLERANCE_REARTH)
    return df


def add_rocky_curve(ax):
    if not DRAW_ROCKY_CURVE:
        return
    m_ref, r_ref = load_rocky_reference_curve()
    m_min = max(NOTEBOOK_XLIM[0], np.nanmin(m_ref))
    m_max = min(NOTEBOOK_XLIM[1], np.nanmax(m_ref))
    if m_max <= m_min:
        return
    m_line = np.linspace(m_min, m_max, 500)
    r_line = rocky_radius_from_mass(m_line, m_ref, r_ref)
    ax.plot(m_line, r_line, linestyle="--", linewidth=1.6, color="black", alpha=0.92, zorder=3)


# ============================================================
# Plot helpers
# ============================================================


def log_flux_values(df: pd.DataFrame):
    flux = pd.to_numeric(df["flux_p"], errors="coerce").clip(lower=1e-12)
    return np.log10(flux.to_numpy(dtype=float))


def compute_shared_color_limits(*dfs, fallback=(-1.0, 2.0)):
    vals = []
    for df in dfs:
        if len(df) > 0:
            vals.append(log_flux_values(df))
    if not vals:
        return fallback
    finite = np.concatenate(vals)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return fallback
    lo, hi = COLOR_PERCENTILES
    vmin = np.nanpercentile(finite, lo)
    vmax = np.nanpercentile(finite, hi)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return fallback
    if np.isclose(vmin, vmax):
        vmin -= 0.1
        vmax += 0.1
    return vmin, vmax


def make_insolation_panels(df: pd.DataFrame):
    return [
        (r"I < 10 $I_\oplus$", df[df["flux_p"] < 10].copy(), "lt10"),
        (r"I < 50 $I_\oplus$", df[df["flux_p"] < 50].copy(), "lt50"),
        (r"I > 50 $I_\oplus$", df[df["flux_p"] > 50].copy(), "gt50"),
    ]


def setup_axis(ax):
    ax.set_xlim(*NOTEBOOK_XLIM)
    ax.set_ylim(*NOTEBOOK_YLIM)
    ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=10)
    ax.set_ylabel(r"Radius [$R_\oplus$]", fontsize=10)
    ax.grid(True, alpha=0.28, linestyle="--")


def visible_in_zoom(df: pd.DataFrame) -> pd.Series:
    return (
        (df["mass_p"] >= NOTEBOOK_XLIM[0])
        & (df["mass_p"] <= NOTEBOOK_XLIM[1])
        & (df["radius_p"] >= NOTEBOOK_YLIM[0])
        & (df["radius_p"] <= NOTEBOOK_YLIM[1])
    )


def label_below_curve_points(ax, df: pd.DataFrame, group_name: str):
    if group_name not in LABEL_BELOW_CURVE_FOR:
        return
    label_df = df[(df["below_rocky_curve"]) & visible_in_zoom(df)].copy()
    for _, row in label_df.iterrows():
        label = str(row.get("planet_label", "")).strip()
        if label == "" or label.lower() == "nan":
            label = str(row.name)
        ax.text(
            row["mass_p"] + LABEL_DX,
            row["radius_p"] + LABEL_DY,
            label,
            fontsize=LABEL_FONT_SIZE,
            alpha=0.78,
            clip_on=True,
            zorder=6,
        )


def draw_errorbars(ax, df: pd.DataFrame):
    if not DRAW_ERRORBARS or len(df) == 0:
        return
    err_df = df[df.get("has_any_errorbar", pd.Series(False, index=df.index))].copy()
    if len(err_df) == 0:
        return
    xerr, yerr = errorbar_arrays(err_df)
    if xerr is None and yerr is None:
        return
    ax.errorbar(
        err_df["mass_p"],
        err_df["radius_p"],
        xerr=xerr,
        yerr=yerr,
        fmt="none",
        ecolor=ERRORBAR_COLOR,
        elinewidth=ERRORBAR_LINEWIDTH,
        alpha=ERRORBAR_ALPHA,
        capsize=ERRORBAR_CAPSIZE,
        zorder=1,
    )


def scatter_panel(ax, df: pd.DataFrame, group_name: str, bin_title: str, vmin, vmax, point_size):
    setup_axis(ax)
    add_rocky_curve(ax)

    if len(df) == 0:
        ax.text(0.5, 0.5, "No planets", transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.set_title(f"{group_name}\n{bin_title}\nN = 0", fontsize=10)
        return None

    draw_errorbars(ax, df)

    sc = ax.scatter(
        df["mass_p"],
        df["radius_p"],
        c=log_flux_values(df),
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        s=point_size,
        alpha=POINT_ALPHA,
        edgecolors="none",
        zorder=2,
    )

    label_below_curve_points(ax, df, group_name)
    n_err = int(df.get("has_any_errorbar", pd.Series(False, index=df.index)).sum())
    n_below = int(df.get("below_rocky_curve", pd.Series(False, index=df.index)).sum())
    ax.set_title(f"{group_name}\n{bin_title}\nN = {len(df)}; bars = {n_err}; below = {n_below}", fontsize=10)
    return sc


# ============================================================
# Saving summaries and plots
# ============================================================


def save_label_lists(nasa: pd.DataFrame, dace: pd.DataFrame, nasa_tag: str):
    label_outputs = []
    groups = [
        (f"NASA_{nasa_tag}", str(nasa["comparison_group"].iloc[0]) if len(nasa) else f"NASA_{nasa_tag}", nasa),
        ("DACE", str(dace["comparison_group"].iloc[0]) if len(dace) else "DACE", dace),
    ]

    for short_name, group_name, df in groups:
        keep_cols = [
            "planet_label", "mass_p", "mass_err_minus", "mass_err_plus", "mass_rel_uncertainty",
            "radius_p", "radius_err_minus", "radius_err_plus", "radius_rel_uncertainty",
            "flux_p", "rocky_radius_ref", "radius_excess_to_rocky", "radius_ratio_to_rocky",
            "below_rocky_curve",
        ]
        extra_cols = [
            c for c in [
                "host_name", "comparison_group", "insolation_source", "discovery_facility",
                "mass_limit_flag", "radius_limit_flag", "insolation_limit_flag",
                "mass_quality_reason", "radius_quality_reason", "mass_reference", "radius_reference",
            ] if c in df.columns
        ]

        sub_all = df[df["below_rocky_curve"]].copy()
        all_path = OUT_DIR / f"{short_name}_below_rocky_curve_all_bins_STRICT_after_filters.csv"
        sub_all[[c for c in keep_cols + extra_cols if c in sub_all.columns]].to_csv(all_path, index=False)
        label_outputs.append((group_name, "all panels", len(sub_all), all_path))

        for bin_title, sub_panel, bin_key in make_insolation_panels(df):
            sub = sub_panel[sub_panel["below_rocky_curve"]].copy()
            path = OUT_DIR / f"{short_name}_below_rocky_curve_{bin_key}_STRICT_after_filters.csv"
            sub[[c for c in keep_cols + extra_cols if c in sub.columns]].to_csv(path, index=False)
            label_outputs.append((group_name, bin_key, len(sub), path))

    return label_outputs


def save_summary(ppop, nasa, dace, nasa_tag: str):
    rows = []
    for group, df in [("P-Pop", ppop), (f"NASA_{nasa_tag}", nasa), ("DACE", dace)]:
        rows.append({
            "group": group,
            "rows_after_filters": len(df),
            "rows_with_mass_errorbar": int(df.get("has_mass_errorbar", pd.Series(False, index=df.index)).sum()),
            "rows_with_radius_errorbar": int(df.get("has_radius_errorbar", pd.Series(False, index=df.index)).sum()),
            "rows_with_any_errorbar": int(df.get("has_any_errorbar", pd.Series(False, index=df.index)).sum()),
            "below_rocky_curve": int(df.get("below_rocky_curve", pd.Series(False, index=df.index)).sum()),
            "median_mass_rel_uncertainty": df.get("mass_rel_uncertainty", pd.Series(np.nan, index=df.index)).median(),
            "median_radius_rel_uncertainty": df.get("radius_rel_uncertainty", pd.Series(np.nan, index=df.index)).median(),
        })
    summary = pd.DataFrame(rows)
    path = OUT_DIR / f"summary_after_filters_{nasa_tag}.csv"
    summary.to_csv(path, index=False)
    print(f"\nSummary after filters ({nasa_tag}):")
    print(summary.to_string(index=False))
    return path


def make_3x3_figure(ppop: pd.DataFrame, nasa: pd.DataFrame, dace: pd.DataFrame, nasa_tag: str, nasa_label: str):
    rows = [
        ("P-Pop detected", ppop, POINT_SIZE_PPOP),
        (nasa_label, nasa, POINT_SIZE_NASA),
        (str(dace["comparison_group"].iloc[0]) if len(dace) else "DACE", dace, POINT_SIZE_DACE),
    ]

    vmin, vmax = compute_shared_color_limits(ppop, nasa, dace)
    fig, axes = plt.subplots(3, 3, figsize=(18, 13.5), constrained_layout=True)
    colorbar_scatter = None

    counts = []
    for i, (group_name, df, point_size) in enumerate(rows):
        for j, (bin_title, df_sub, bin_key) in enumerate(make_insolation_panels(df)):
            sc = scatter_panel(
                axes[i, j],
                df_sub,
                group_name,
                bin_title,
                vmin,
                vmax,
                point_size,
            )
            if colorbar_scatter is None and sc is not None:
                colorbar_scatter = sc
            counts.append({
                "group": group_name,
                "bin": bin_title,
                "count_after_filters": len(df_sub),
                "points_with_errorbars": int(df_sub.get("has_any_errorbar", pd.Series(False, index=df_sub.index)).sum()),
                "below_rocky_curve": int(df_sub.get("below_rocky_curve", pd.Series(False, index=df_sub.index)).sum()),
            })

    if colorbar_scatter is not None:
        cbar = fig.colorbar(colorbar_scatter, ax=axes.ravel().tolist(), location="right", shrink=0.93, pad=0.015)
        cbar.set_label(r"log$_{10}$(Insolation Flux [$I_\oplus$])", fontsize=11)

    strict_text = "strict NASA M-R quality filter ON" if NASA_REQUIRE_STRICT_MR_QUALITY else "strict NASA M-R quality filter OFF"
    fig.suptitle(
        "P-Pop, NASA PSCompPars, and DACE split by insolation\n"
        f"NASA version: {nasa_label}; {strict_text}; mass uncertainty <= {MAX_MASS_REL_UNCERTAINTY:.0%}; "
        f"radius uncertainty <= {MAX_RADIUS_REL_UNCERTAINTY:.0%}",
        fontsize=14,
    )

    png = OUT_DIR / f"ppop_nasa_{nasa_tag}_dace_3x3_insolation_pscomppars_flags.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    counts_df = pd.DataFrame(counts)
    counts_path = OUT_DIR / f"ppop_nasa_{nasa_tag}_dace_3x3_insolation_counts_pscomppars_flags.csv"
    counts_df.to_csv(counts_path, index=False)

    print(f"\nSaved {nasa_tag} 3x3 PNG:")
    print(png)
    print("Saved counts:")
    print(counts_path)
    return png, counts_path


# ============================================================
# Main
# ============================================================


def main():
    print("Project root:", ROOT)
    print("NASA cache path:", NASA_PSCOMPPARS_FLAGS_CACHE)
    print("Output directory:", OUT_DIR)
    print(f"Strict NASA M-R quality: {NASA_REQUIRE_STRICT_MR_QUALITY}")
    print(f"Uncertainty cuts: mass <= {MAX_MASS_REL_UNCERTAINTY:.0%}, radius <= {MAX_RADIUS_REL_UNCERTAINTY:.0%}")

    # The NASA download/caching happens here before plotting.
    _ = load_nasa_pscomppars_with_flags()

    ppop = add_rocky_diagnostics(load_ppop_detected())
    dace = add_rocky_diagnostics(load_dace_sample())

    ppop.to_csv(OUT_DIR / "plot_data_ppop_detected_after_filters.csv", index=False)
    dace.to_csv(OUT_DIR / "plot_data_dace_after_filters.csv", index=False)

    outputs = []
    all_label_outputs = []

    for facility_filter, tag, label in NASA_VERSIONS:
        print("\n" + "=" * 80)
        print(f"NASA VERSION: {tag}")
        nasa = add_rocky_diagnostics(load_nasa_observed(facility_filter, tag))
        nasa["comparison_group"] = label
        nasa.to_csv(OUT_DIR / f"plot_data_nasa_{tag}_STRICT_after_filters.csv", index=False)

        save_summary(ppop, nasa, dace, tag)
        outputs.append(make_3x3_figure(ppop, nasa, dace, tag, label))
        all_label_outputs.extend(save_label_lists(nasa, dace, tag))

    print("\nSaved PNG outputs:")
    for png, _ in outputs:
        print(png)

    print("\nSaved below-rocky label lists:")
    for group_name, bin_key, n, path in all_label_outputs:
        print(f"{group_name} [{bin_key}]: {n} below-curve planets -> {path}")

    print("\nDone.")
    print("Caveman check:")
    print("  NASA warning flags downloaded.")
    print("  Upper-limit/calculated/no-error NASA masses removed before rocky-curve plot.")
    print("  Kepler-409 b / Kepler-106 d style mistakes should not enter the plotted NASA strict sample.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nSCRIPT FAILED")
        print(f"{type(exc).__name__}: {exc}")
        sys.exit(1)

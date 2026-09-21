"""The three detector recovery panels in one figure (recovery_3x1.png)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools.paths import ( KOI_CUMULATIVE_CSV, EXOFOP_TOI_CSV, PAPER_FIGURES_DIR,
                         CALIBRATION_DIR, TESS_DATA_DIR, KEPLER_DATA_DIR)
from science.catalogs import nasa_tap_url, read_nasa_csv
from science.physics import infer_stellar_type
from plotting.figure_style import PAPER_STYLE

from science.telescopes.kepler.detection_model import KeplerData
from science.telescopes.tess.detection_model import TESSData
from science.telescopes.rv.detection_model import RVData
from science.telescopes.tess import download_tce_stats
from science.telescopes.tess.build_reference_data import MAX_SECTOR

OUT_DIR = Path(CALIBRATION_DIR) / "recovery_3x1"
OUT_DIR.mkdir(parents=True, exist_ok=True)
KEPLER_DIR = TESS_DIR = RV_DIR = OUT_DIR
KEPLER_CACHE = OUT_DIR / "koi_stellar_cached.csv"
TESS_CACHE = OUT_DIR / "toi_cached.csv"
DEPTH_ERR_CACHE = OUT_DIR / "koi_depth_errors.csv"
PAPER_FIG_DIR = Path(PAPER_FIGURES_DIR)
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOAD_NASA_DATA = False          # True refreshes the catalogues from the NASA archive
MES_THRESHOLD = 7.1
SNR_THRESHOLD = 7.1
PLANET_DISPOSITIONS = ("CONFIRMED", "CANDIDATE")   # false positives are not planets to recover
CDPP_DIR = Path(TESS_DATA_DIR) / "CDPP"
MAX_CDPP_SECTOR = 106
PANEL_SIZE = (7.0, 6.0)             # each panel, inches

RVAMP_CACHE = Path(KEPLER_DATA_DIR) / "NASA" / "NASA_PSCompPars_rvamp_calibration.csv"
ESO_TARGETS_CACHE = Path(KEPLER_DATA_DIR) / "NASA" / "eso_harps_nirps_targets.csv"
ESO_TAP = "http://archive.eso.org/tap_obs/sync"
ESO_MATCH_ARCSEC = 10.0
ESO_MIN_EXPOSURES = 10
RV_ARGS = argparse.Namespace(instrument="HARPS", n_obs=100, snr_threshold=5.0)


# ── Kepler ────────────────────────────────────────────────────────────────────

CDPP_COLS = [
    "rrmscdpp01p5", "rrmscdpp02p0", "rrmscdpp02p5", "rrmscdpp03p0",
    "rrmscdpp03p5", "rrmscdpp04p5", "rrmscdpp05p0", "rrmscdpp06p0",
    "rrmscdpp07p5", "rrmscdpp09p0", "rrmscdpp10p5", "rrmscdpp12p0",
    "rrmscdpp12p5", "rrmscdpp15p0",
]

def kepler_load(redownload: bool = False) -> pd.DataFrame:
    """KOI cumulative + stellar CDPP. Reads the copy in data/; downloads only when
    DOWNLOAD_NASA_DATA (or redownload) is set, and saves what it fetches back to data/.
    """
    local = Path(KOI_CUMULATIVE_CSV)
    if local.exists() and not (DOWNLOAD_NASA_DATA or redownload):
        print(f"Loading local KOI table: {local}")
        df = read_nasa_csv(local)
        missing = [c for c in ("koi_max_mult_ev", "koi_num_transits")
                   if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"{local.name} lacks {missing}. Set DOWNLOAD_NASA_DATA = True to "
                "refresh it from the NASA Exoplanet Archive.")
        return df

    if not (DOWNLOAD_NASA_DATA or redownload):
        raise RuntimeError(
            f"{local} not found. Set DOWNLOAD_NASA_DATA = True to fetch the KOI "
            "cumulative table from the NASA Exoplanet Archive.")

    cdpp_select = ", ".join(f"s.{c}" for c in CDPP_COLS)
    query = f"""
    SELECT
        c.kepid, c.kepoi_name, c.koi_disposition, c.koi_pdisposition,
        c.koi_period, c.koi_prad, c.koi_sma, c.koi_insol,
        c.koi_depth, c.koi_depth_err1, c.koi_depth_err2, c.koi_duration, c.koi_incl,
        c.koi_steff, c.koi_srad, c.koi_smass, c.koi_kepmag,
        c.koi_max_sngle_ev, c.koi_max_mult_ev, c.koi_model_snr, c.koi_num_transits,
        c.koi_tce_delivname,
        s.dataspan, s.dutycycle,
        {cdpp_select}
    FROM cumulative c
    JOIN keplerstellar s ON c.kepid = s.kepid
    WHERE c.koi_period IS NOT NULL
      AND c.koi_prad   IS NOT NULL
      AND c.koi_srad   IS NOT NULL
      AND c.koi_depth  IS NOT NULL
      AND c.koi_duration IS NOT NULL
    """
    # NASA retired the one-row-per-target q1_q17_dr25_stellar table; keplerstellar
    # holds the same rrmscdpp* columns but with one row per (target, data release),
    # so the JOIN multiplies each KOI.  Collapse back to one row per KOI by keeping
    # the longest-baseline (largest dataspan) row that carries a valid CDPP value.
    print("Downloading KOI + stellar CDPP from NASA ExoplanetArchive (JOIN query)...")
    df = pd.read_csv(nasa_tap_url(query), low_memory=False)
    n_raw = len(df)
    df["dataspan"] = pd.to_numeric(df["dataspan"], errors="coerce")
    df["rrmscdpp07p5"] = pd.to_numeric(df["rrmscdpp07p5"], errors="coerce")
    key = "kepoi_name" if "kepoi_name" in df.columns else "kepid"
    df = (df[df["rrmscdpp07p5"].notna()]
          .sort_values("dataspan", ascending=False)
          .drop_duplicates(subset=key, keep="first")
          .reset_index(drop=True))
    print(f"  Deduped keplerstellar rows: {n_raw:,} -> {len(df):,} (one per {key})")
    df.to_csv(KOI_CUMULATIVE_CSV, index=False)
    print(f'Saved to: {KOI_CUMULATIVE_CSV}')
    print(f"  Saved: {KEPLER_CACHE}  ({len(df):,} rows)")
    return df

def attach_depth_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Add koi_depth_err1/2, which the local KOI table predates. One small cached TAP query,
    keyed by KOI name; the figure's error bars are the only thing that needs them."""
    if {"koi_depth_err1", "koi_depth_err2"}.issubset(df.columns):
        return df
    if DEPTH_ERR_CACHE.exists():
        err = pd.read_csv(DEPTH_ERR_CACHE)
    else:
        print("Downloading KOI depth uncertainties from NASA Exoplanet Archive...")
        err = pd.read_csv(nasa_tap_url(
            "select kepoi_name, koi_depth_err1, koi_depth_err2 from cumulative "
            "where koi_depth is not null"), low_memory=False)
        err.to_csv(DEPTH_ERR_CACHE, index=False)
        print(f"  Saved: {DEPTH_ERR_CACHE}  ({len(err):,} rows)")
    merged = df.merge(err.drop_duplicates("kepoi_name"), on="kepoi_name", how="left")
    have = pd.to_numeric(merged["koi_depth_err1"], errors="coerce").notna().sum()
    print(f"Depth uncertainties matched: {have:,}/{len(merged):,}")
    return merged

def kepler_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to DR25 TCEs, set tran_flag=1 (all KOIs are transit-like events)."""
    numeric_cols = [
        "koi_period", "koi_prad", "koi_sma", "koi_insol",
        "koi_depth", "koi_depth_err1", "koi_depth_err2", "koi_duration", "koi_incl", "koi_steff",
        "koi_srad", "koi_smass", "koi_kepmag",
        "koi_max_sngle_ev", "koi_max_mult_ev", "koi_model_snr", "koi_num_transits",
        "dataspan", "dutycycle",
    ] + CDPP_COLS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Restrict to DR25 KOIs when available.
    if "koi_tce_delivname" in df.columns:
        is_dr25 = df["koi_tce_delivname"].astype(str).str.contains("Q1_Q17_DR25", case=False, na=False)
        if is_dr25.any():
            before = len(df); df = df[is_dr25].copy()
            print(f"DR25 filter: {before:,} -> {len(df):,}")

    required = ["koi_period", "koi_prad", "koi_srad", "koi_depth", "koi_duration"]
    df = df.dropna(subset=required).copy()
    df = df[(df["koi_period"] > 0) & (df["koi_prad"] > 0) & (df["koi_srad"] > 0)].copy()
    df["tran_flag"] = 1   # KOIs are already transit-like events
    print(f"Valid KOI rows: {len(df):,}")
    return df

def kepler_run(df: pd.DataFrame) -> pd.DataFrame:
    """Run KeplerData; it will use real rrmscdpp* columns when present."""
    kd = KeplerData(
        df,
        source="koi",
        validate_for_detection=True,
        min_transits=3,
        mes_threshold=MES_THRESHOLD,
        kepler_mag_limit=16.0,
        fallback_cdpp_ppm=50.0,    # better calibrated fallback (was 100)
        use_kepmag_cdpp_fallback=True,
        use_observed_transit_flag_for_nasa=True,
        use_observed_transit_depth_for_nasa=True,
        assume_bright_if_kepmag_missing_for_nasa=False,
        estimate_missing_semimajor_axis=True,
    )
    out = kd.determine_detectable()
    if "kepoi_name" not in out.columns and "planet_name" in out.columns:
        out["kepoi_name"] = out["planet_name"]
    out["toy_over_koi_mes"] = (
        pd.to_numeric(out["kepler_mes"], errors="coerce") /
        pd.to_numeric(out["koi_max_mult_ev"], errors="coerce").replace(0, np.nan)
    )
    # Tag which rows used real CDPP vs fallback.
    out["used_real_cdpp"] = out.get("kepler_cdpp_source", pd.Series("unknown", index=out.index)).str.contains("rrmscdpp", na=False)
    return out

def draw_kepler(ax, out: pd.DataFrame) -> float:
    """One panel: model vs official MES for Kepler planets, coloured by detection."""
    is_planet = out["koi_disposition"].astype(str).str.upper().isin(PLANET_DISPOSITIONS)

    # MES to MES comparison
    official = pd.to_numeric(out["koi_max_mult_ev"], errors="coerce")
    model = pd.to_numeric(out["kepler_mes"], errors="coerce")
    x_label, y_label, title = "Kepler MES", "Model MES", "Kepler model planet recovery"

    keep = official.gt(0) & model.gt(0) & is_planet
    official, model = official[keep], model[keep]
    passed = out.loc[keep, "detected"].astype(bool).to_numpy()

    # Error bars on official value (from depth uncertainty)
    depth = pd.to_numeric(out.loc[keep, "observed_transit_depth_ppm"], errors="coerce")
    depth_err = pd.concat([
        pd.to_numeric(out.loc[keep, "koi_depth_err1"], errors="coerce").abs(),
        pd.to_numeric(out.loc[keep, "koi_depth_err2"], errors="coerce").abs(),
    ], axis=1).mean(axis=1)
    xerr = (official * (depth_err / depth)).fillna(0.0).clip(lower=0.0)

    for mask, colour, label in [(passed, "#2060c0", "Model detects"),
                                (~passed, "#d95f02", "Model misses")]:
        ax.errorbar(official[mask], model[mask], xerr=xerr[mask], fmt="o", ms=5.5, alpha=0.55,
                    color=colour, ecolor=colour, elinewidth=0.7, capsize=0, linestyle="none",
                    label=label)
    lims = [official.min() * 0.7, official.max() * 1.5]
    ax.plot(lims, lims, "k--", lw=1.2, label="Equal")
    ax.axhline(MES_THRESHOLD, color="0.4", ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.tick_params(labelsize=22)
    ax.legend(loc="upper left", fontsize=19, markerscale=2.2)
    ax.text(0.97, 0.05, f"Overall recovery = {passed.mean():.0%}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=20)
    print(f"  Recovered {passed.mean():.0%} of {len(official):,} confirmed and candidate DR25 KOIs")
    return float(passed.mean())


# ── TESS ──────────────────────────────────────────────────────────────────────

def searched_sectors(sectors_text, source_text):
    """Sectors a SPOC multi-sector run searched for this TOI: the TOI's listed sectors inside the
    run's range (Source 'spoc-s01-s69-...'). None for QLP, single-sector 'spoc', or runs reaching
    past the last sector with a CDPP table.
    """
    m = re.match(r"spoc-s(\d+)-s(\d+)", str(source_text).lower())
    if not m or int(m.group(2)) > MAX_CDPP_SECTOR:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    secs = [int(s) for s in re.split(r"[,;\s]+", str(sectors_text)) if s.strip().isdigit()]
    secs = [s for s in secs if lo <= s <= hi]
    return ";".join(map(str, secs)) if secs else None

EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"

def tess_load(redownload: bool = False) -> pd.DataFrame:
    """ExoFOP TOI table. Reads the copy in data/; downloads only when DOWNLOAD_NASA_DATA
    (or redownload) is set, and saves what it fetches back to data/.
    """
    local = Path(EXOFOP_TOI_CSV)
    if local.exists() and not (DOWNLOAD_NASA_DATA or redownload):
        print(f"Loading local TOI table: {local}")
        df = read_nasa_csv(local)
        if "Period (days)" not in df.columns:
            raise RuntimeError(
                f"{local.name} is not an ExoFOP TOI table (no 'Period (days)' "
                "column). Set DOWNLOAD_NASA_DATA = True to refresh it.")
        return df

    if not (DOWNLOAD_NASA_DATA or redownload):
        raise RuntimeError(
            f"{local} not found. Set DOWNLOAD_NASA_DATA = True to fetch the TOI "
            "table from ExoFOP.")

    print(f"Downloading ExoFOP TESS TOI catalog...")
    try:
        df = pd.read_csv(EXOFOP_TOI_URL, low_memory=False)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot download ExoFOP TOI. Check internet connection.\n{exc}"
        ) from exc
    df.to_csv(TESS_CACHE, index=False)
    print(f"  Saved: {TESS_CACHE}  ({len(df):,} rows)")
    return df

def tess_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Rename ExoFOP TOI columns (TIC ID, Period, Duration, Depth, Stellar Radius/Teff,
    TESS Mag, TFOPWG Disposition, SNR, ...) to the internal names TESSData expects.
    """
    df = df.copy()
    # Normalise column names.
    df.columns = df.columns.str.strip()

    col_map = {
        "TIC ID":                "ticid",
        "TOI":                   "pl_toi",
        "Period (days)":         "p_orb",
        "Duration (hours)":      "observed_duration_hr",
        "Depth (ppm)":           "observed_depth_ppm",
        "Depth (ppm) err":       "observed_depth_err_ppm",
        "Planet Radius (R_Earth)": "radius_p",
        "Stellar Radius (R_Sun)":"radius_s",
        "Stellar Eff Temp (K)":  "teff_s",
        # Map to the internal "tmag" name that TESSData ingests and proxies from;
        # mapping straight to "tess_tmag" gets clobbered (TESSData rebuilds
        # tess_tmag from "tmag", overwriting it with NaN -> brightness gate fails).
        "TESS Mag":              "tmag",
        "RA":                    "ra",
        "Dec":                   "dec",
        "TFOPWG Disposition":    "tfopwg_disp",
        "Planet SNR":            "official_snr",   # ExoFOP renamed "SNR" -> "Planet SNR"
        "SNR":                   "official_snr",
        "Stellar Distance (pc)": "distance_s",
        "Stellar log(g) (cm/s^2)": "logg_s",
        "Stellar Mass (M_Sun)":  "mass_s",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    numeric_cols = ["p_orb", "observed_duration_hr", "observed_depth_ppm", "observed_depth_err_ppm", "radius_p",
                    "radius_s", "teff_s", "tmag", "official_snr", "distance_s",
                    "mass_s", "ra", "dec"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # All TOIs are observed transiting events.
    df["tran_flag"] = 1

    # Derive stellar radius from radius_p and depth if radius_s missing.
    if "radius_s" not in df.columns or df["radius_s"].isna().all():
        if "radius_p" in df.columns and "observed_depth_ppm" in df.columns:
            depth_frac = pd.to_numeric(df["observed_depth_ppm"], errors="coerce") / 1e6
            rp = pd.to_numeric(df["radius_p"], errors="coerce")
            rs_from_depth = rp / np.sqrt(depth_frac.clip(lower=1e-8))
            df["radius_s"] = rs_from_depth / 109.076  # R_earth to R_sun
            df["radius_s_source"] = "inferred_from_depth"

    # Estimate semimajor axis from period and stellar mass.
    if "mass_s" not in df.columns:
        df["mass_s"] = 1.0
    if "semimajor_p" not in df.columns:
        p_yr = df["p_orb"] / 365.25
        df["semimajor_p"] = (df["mass_s"].fillna(1.0) * p_yr ** 2) ** (1.0 / 3.0)

    # Estimate radius_p from depth if missing.
    if "radius_p" not in df.columns or df["radius_p"].isna().mean() > 0.5:
        if "observed_depth_ppm" in df.columns and "radius_s" in df.columns:
            depth_frac = pd.to_numeric(df["observed_depth_ppm"], errors="coerce") / 1e6
            rs_rearth = pd.to_numeric(df["radius_s"], errors="coerce") * 109.076
            df["radius_p"] = rs_rearth * np.sqrt(depth_frac.clip(lower=0))

    # Inclination unknown: every TOI transits, and TESSData solves b from the measured T14.
    df["tran_flag"] = 1

    required = ["p_orb", "radius_s", "radius_p", "tmag"]
    df = df.dropna(subset=[c for c in required if c in df.columns]).copy()
    df = df[df["p_orb"] > 0].copy()
    # Drop rows without official SNR (needed for calibration).
    if "official_snr" in df.columns:
        df = df.dropna(subset=["official_snr"]).copy()
        df = df[df["official_snr"] > 0].copy()

    df["tess_sectors"] = [searched_sectors(s, src) for s, src in zip(df["Sectors"], df["Source"])]
    df = df.dropna(subset=["tess_sectors"]).copy()
    print(f"Valid SPOC multi-sector TOI rows (with official SNR): {len(df):,}")
    return df

def attach_official_mes(df: pd.DataFrame) -> pd.DataFrame:
    """The MES the pipeline detected on, from the DV run named in each TOI's Source. ExoFOP
    publishes only 'Planet SNR', which is neither the MES nor the model SNR, and MES depends on how
    many sectors a run stacked, so it has to come from that TOI's own run."""
    span = df["Source"].astype(str).str.extract(r"spoc-s(\d+)-s(\d+)").astype(float)
    df = df.copy()
    df["_span"] = list(zip(span[0], span[1]))
    wanted = {(int(a), int(b)) for a, b in df["_span"] if pd.notna(a) and pd.notna(b)}
    tables = download_tce_stats.fetch(wanted)
    df["official_mes"] = np.nan
    for sp, path in tables.items():
        tce = pd.read_csv(path, low_memory=False)
        rows = df["_span"].apply(lambda s, sp=sp: (pd.notna(s[0]) and (int(s[0]), int(s[1])) == sp))
        sub = df[rows]
        if sub.empty:
            continue
        j = sub[["ticid", "p_orb"]].reset_index().merge(tce, on="ticid", how="inner")
        j["_dp"] = (j["p_orb"] - j["tce_period"]).abs() / j["p_orb"]
        j = j[j["_dp"] < 0.01].sort_values("_dp").drop_duplicates("index", keep="first")
        df.loc[j["index"], "official_mes"] = j["tce_max_mult_ev"].to_numpy()
    n = df["official_mes"].notna().sum()
    print(f"TOIs matched to their run's MES: {n:,}/{len(df):,}")
    return df.drop(columns="_span")

def tess_run(df: pd.DataFrame) -> pd.DataFrame:
    """Run TESSData on each TOI's searched 2-min sectors with its own per-TIC SPOC CDPP."""
    if not CDPP_DIR.exists():
        raise FileNotFoundError(f"SPOC CDPP CSVs not found in {CDPP_DIR}")
    # SPOC 2-min runs only search sectors where the star was a 2-min target, i.e. is in that
    # sector's CDPP table; the TOI's Sectors column also lists FFI-only sectors.
    cdpp = TESSData._load_cdpp_tables(TESSData.__new__(TESSData), CDPP_DIR)
    two_min = set(zip(cdpp["ticid"].astype("Int64").astype(int), cdpp["sector"].astype(int)))
    tic = pd.to_numeric(df["ticid"], errors="coerce")
    df = df.copy()
    df["tess_sectors"] = [
        ";".join(x for x in str(secs).split(";") if x and pd.notna(t) and (int(t), int(x)) in two_min) or None
        for secs, t in zip(df["tess_sectors"], tic)]
    df = df.dropna(subset=["tess_sectors"])
    print(f"TOIs with at least one searched 2-min sector: {len(df):,}")

    out = TESSData(df, source="nasa", min_transits=3, snr_threshold=SNR_THRESHOLD, tmag_limit=16.0,
                   use_catalog_sectors=True, phase_mode="expected", cdpp_dir=CDPP_DIR,
                   validate_for_detection=True).determine_detectable()
    out["toy_over_official_snr"] = (
        pd.to_numeric(out["tess_snr"], errors="coerce") /
        pd.to_numeric(out.get("official_snr", pd.Series(np.nan, index=out.index)), errors="coerce").replace(0, np.nan)
    )
    return out

def draw_tess(ax, out: pd.DataFrame) -> float:
    """One panel: model vs official MES for TESS TOIs, coloured by detection."""
    # Load official MES and model SNR
    official = pd.to_numeric(out.get("official_mes", pd.Series(np.nan, index=out.index)), errors="coerce")
    model = pd.to_numeric(out["tess_snr"], errors="coerce")
    x_label, y_label, title = "TESS MES", "Model SNR", "TESS model planet recovery"

    keep = official.gt(0) & model.gt(0)
    official, model = official[keep], model[keep]
    passed = out.loc[keep, "tess_detected"].astype(bool).to_numpy()

    # Error bars on official value (from depth uncertainty)
    depth = pd.to_numeric(out.loc[keep, "observed_depth_ppm"], errors="coerce")
    depth_err = pd.to_numeric(out.loc[keep, "observed_depth_err_ppm"], errors="coerce")
    xerr = (official * (depth_err / depth)).fillna(0.0).clip(lower=0.0)

    for mask, colour, label in [(passed, "#2060c0", "Model detects"),
                                (~passed, "#d95f02", "Model misses")]:
        ax.errorbar(official[mask], model[mask], xerr=xerr[mask], fmt="o", ms=5.5, alpha=0.55,
                    color=colour, ecolor=colour, elinewidth=0.7, capsize=0, linestyle="none",
                    label=label)
    lims = [official.min() * 0.7, official.max() * 1.5]
    ax.plot(lims, lims, "k--", lw=1.2, label="Equal")
    ax.axhline(SNR_THRESHOLD, color="0.4", ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.tick_params(labelsize=22)
    ax.legend(loc="upper left", fontsize=19, markerscale=2.2)
    ax.text(0.97, 0.05, f"Overall recovery = {passed.mean():.0%}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=20)
    n_tois = len(official)
    print(f"  Recovered {passed.mean():.0%} of {n_tois:,} TESS TOIs (matched to SPOC MES)")
    return float(passed.mean())


# ── Radial velocity ───────────────────────────────────────────────────────────

RVAMP_QUERY = """
SELECT pl_name, hostname, discoverymethod, tran_flag,
       pl_orbper, pl_orbeccen, pl_orbincl,
       pl_rvamp, pl_rvamperr1, pl_rvamperr2,
       pl_bmasse, pl_bmassprov, pl_msinie,
       pl_rade, pl_insol,
       st_mass, st_rad, st_teff, st_lum, sy_dist, sy_vmag
FROM pscomppars
WHERE pl_rvamp IS NOT NULL AND pl_rvamp > 0
  AND pl_orbper IS NOT NULL AND st_mass IS NOT NULL
"""

def load_rvamp_sample() -> pd.DataFrame:
    if RVAMP_CACHE.exists():
        print(f"Loading published-K sample from cache: {RVAMP_CACHE.name}")
        df = pd.read_csv(RVAMP_CACHE)
    else:
        print("Downloading PSCompPars planets with published pl_rvamp ...")
        df = pd.read_csv(nasa_tap_url(RVAMP_QUERY))
        RVAMP_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RVAMP_CACHE, index=False)
        print(f"Saved cache: {RVAMP_CACHE}")
    for c in df.columns:
        if c not in ("pl_name", "hostname", "discoverymethod", "pl_bmassprov"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  Published-K sample: {len(df):,} planets")
    return df

def model_k_from_published(df: pd.DataFrame) -> pd.Series:
    """K from the planet's published parameters, using M sin i when present
    (else best mass; most of those planets transit, so i ~ 90 deg).
    """
    msini = df["pl_msinie"].where(df["pl_msinie"].notna(), df["pl_bmasse"])
    m_jup = msini / 317.828
    p_yr = df["pl_orbper"] / 365.25
    ecc = df["pl_orbeccen"].fillna(0.0).clip(0, 0.95)
    return (28.4329 * m_jup * df["st_mass"] ** (-2 / 3) * p_yr ** (-1 / 3)
            / np.sqrt(1 - ecc ** 2))

ESO_QUERY = ("select instrument,object,min(ra) as ra,min(dec) as dec,count(*) as n_exp "
             "from dbo.raw where instrument in ('HARPS','NIRPS') and dp_cat='SCIENCE' "
             "group by instrument,object")

def load_eso_targets() -> pd.DataFrame:
    """Every star HARPS or NIRPS has observed, from the ESO science archive (cached)."""
    if not ESO_TARGETS_CACHE.exists():
        url = f"{ESO_TAP}?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY={quote(ESO_QUERY)}"
        print("Downloading HARPS/NIRPS target list from the ESO archive ...")
        frame = pd.read_csv(url)
        ESO_TARGETS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(ESO_TARGETS_CACHE, index=False)
    return pd.read_csv(ESO_TARGETS_CACHE)

def observed_by_harps_nirps(planet_names: pd.Series) -> np.ndarray:
    """True where HARPS or NIRPS has pointed at the planet's host, matched on sky position."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from science.catalogs import read_nasa_csv
    from tools.paths import PSCOMPPARS_CSV

    archive = read_nasa_csv(PSCOMPPARS_CSV)[["pl_name", "ra", "dec"]]
    hosts = pd.DataFrame({"pl_name": planet_names.to_numpy()}).merge(archive, on="pl_name", how="left")
    eso = load_eso_targets().dropna(subset=["ra", "dec"])
    # A few archive rows carry parked-telescope coordinates (dec well outside +-90).
    eso = eso[eso["dec"].between(-90, 90) & eso["ra"].between(0, 360)]
    known = hosts["ra"].notna().to_numpy()
    idx_host, idx_eso, _, _ = SkyCoord(eso.ra.to_numpy() * u.deg, eso.dec.to_numpy() * u.deg).search_around_sky(
        SkyCoord(hosts.loc[known, "ra"].to_numpy() * u.deg, hosts.loc[known, "dec"].to_numpy() * u.deg),
        ESO_MATCH_ARCSEC * u.arcsec)
    exposures = np.zeros(len(hosts))
    np.add.at(exposures, np.flatnonzero(known)[idx_host], eso["n_exp"].to_numpy()[idx_eso])
    return exposures >= ESO_MIN_EXPOSURES

def rv_prepare(df: pd.DataFrame, args) -> pd.DataFrame:
    df = df.dropna(subset=["pl_rvamp", "pl_orbper", "st_mass"]).copy()
    df = df[(df["pl_rvamp"] > 0) & (df["pl_msinie"].notna() | df["pl_bmasse"].notna())]
    df["k_model"] = model_k_from_published(df)
    df = df[np.isfinite(df["k_model"]) & (df["k_model"] > 0)].copy()
    df["stype"] = infer_stellar_type(df["st_teff"])
    ratio = df["k_model"] / df["pl_rvamp"]
    print(f"  K calibration sample: {len(df):,}  |  median K_model/K_pub = {ratio.median():.3f}  "
          f"(16-84%: {ratio.quantile(0.16):.3f}-{ratio.quantile(0.84):.3f})")

    # Run the full RV detector on the same planets, for the recovery figure.
    rv = RVData(df.rename(columns={
        "pl_orbper": "p_orb", "pl_orbeccen": "ecc_p", "pl_orbincl": "inc_p",
        "pl_msinie": "msini_p", "pl_bmasse": "mass_p", "pl_rade": "radius_p",
        "st_mass": "mass_s", "st_rad": "radius_s", "st_teff": "teff_s",
        "st_lum": "st_lum_log10", "sy_dist": "distance_s", "sy_vmag": "vmag",
    }), source="pscomppars", instrument=args.instrument if args.instrument.lower() != "best" else "HARPS",
        apply_sini=True, n_obs=args.n_obs, snr_threshold=args.snr_threshold,
        validate_for_detection=False)
    cat = rv.determine_detectable()
    df["rv_pass"] = cat["rv_detected"].to_numpy()
    df["sigma_k_model"] = pd.to_numeric(cat["rv_sigma_K_ms"], errors="coerce").to_numpy()
    df["snr_model"] = pd.to_numeric(cat["rv_snr"], errors="coerce").to_numpy()

    df.to_csv(RV_DIR / "rv_k_calibration_matched_rows.csv", index=False)
    return df

def draw_rv(ax, df: pd.DataFrame) -> float:
    """One panel for both questions: model K against published K (1:1 = right amplitude), with each
    planet coloured by whether the model detects it. Misses are split by whether HARPS or NIRPS has
    actually observed the host (ESO science archive), since those are the instruments modelled; a
    planet only ever measured elsewhere may need a longer or more precise campaign than this one."""
    harps_like = observed_by_harps_nirps(df["pl_name"])

    passed = df["rv_pass"].to_numpy(bool)
    # x error bars are the archive's published K uncertainty; the model K is computed from the
    # published mass, period and stellar mass, so it carries no independent error here.
    err_up = pd.to_numeric(df.get("pl_rvamperr1"), errors="coerce").abs().fillna(0.0)
    err_dn = pd.to_numeric(df.get("pl_rvamperr2"), errors="coerce").abs().fillna(0.0)
    groups = [(passed, "#2060c0", "Model detects"),
              (~passed & harps_like, "#d95f02", "Model misses (HARPS/NIRPS)"),
              (~passed & ~harps_like, "#909090", "Model misses (other RV telescopes)")]
    for mask, colour, label in groups:
        ax.errorbar(df.loc[mask, "pl_rvamp"], df.loc[mask, "k_model"],
                    xerr=np.vstack([err_dn[mask], err_up[mask]]),
                    fmt="o", ms=5.5, alpha=0.55, color=colour, ecolor=colour,
                    elinewidth=0.7, capsize=0, linestyle="none", label=label)
    lims = [0.3, df["pl_rvamp"].quantile(0.999) * 2]
    ax.plot(lims, lims, "k--", lw=1.2, label="Equal")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Published K [m/s]")
    ax.set_ylabel("Model K [m/s]")
    ax.set_title("Radial velocity model planet recovery")
    ax.tick_params(labelsize=22)
    ax.legend(loc="upper left", fontsize=16, markerscale=2.2)
    summary = "\n".join([
        f"Overall recovery (all telescopes) = {passed.mean():.0%}",
        f"Overall recovery (HARPS/NIRPS) = {passed[harps_like].mean():.0%}",
    ])
    ax.text(0.97, 0.04, summary, transform=ax.transAxes, ha="right", va="bottom", fontsize=16)
    print(f"  Recovered {passed.mean():.0%} of {len(df):,} published-K planets; "
          f"{passed[harps_like].mean():.0%} of the {harps_like.sum():,} hosts HARPS/NIRPS observed")
    return float(passed.mean())


# ── Figure ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--redownload", action="store_true")
    args = ap.parse_args()
    plt.rcParams.update(PAPER_STYLE)

    width, height = PANEL_SIZE
    fig, axes = plt.subplots(1, 3, figsize=(3 * width, height), layout="constrained")

    print("Kepler...")
    draw_kepler(axes[0], kepler_run(kepler_prepare(attach_depth_errors(
        kepler_load(redownload=args.redownload)))))
    print("TESS...")
    tess_data = tess_prepare(tess_load(redownload=args.redownload))
    tess_data = attach_official_mes(tess_data)
    draw_tess(axes[1], tess_run(tess_data))
    print("Radial velocity...")
    draw_rv(axes[2], rv_prepare(load_rvamp_sample(), RV_ARGS))

    for ax in axes:
        ax.set_title(ax.get_title().replace(" model planet recovery", ""))
    fig.savefig(OUT_DIR / "recovery_3x1.png", bbox_inches="tight")
    fig.savefig(PAPER_FIG_DIR / "recovery_3x1.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved paper copy: {PAPER_FIG_DIR / 'recovery_3x1.png'}")


if __name__ == "__main__":
    main()

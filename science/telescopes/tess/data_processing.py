"""Catalog and reference-data helpers for the TESS toy detector."""

from __future__ import annotations

from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from science.physics import (
    apparent_bolometric_mag,
    infer_stellar_type,
    semimajor_axis_from_period,
)
from tools.paths import TESS_DATA_DIR, TESS_REFERENCE_DATA_DIR

try:
    from lifesim.core.data import Data
except Exception:  # lets this file import outside the lifesim environment
    Data = None


MAX_SECTOR = 106
GRID_N_LON = 360
GRID_N_SINLAT = 180
OBLIQUITY_DEG = 23.439
TMAG_BIN = 0.5
MIN_STARS_PER_BIN = 10


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
    if {"pl_name", "pl_rade", "pl_orbper"}.issubset(cols):
        return "pscomppars"
    if {"kepoi_name", "koi_prad", "koi_period"}.issubset(cols):
        return "koi"
    return "ppop"


def standardize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    rename = {
        "pl_name": "planet_name", "hostname": "host_name", "tran_flag": "tran_flag",
        "ra": "ra", "dec": "dec", "pl_orbper": "p_orb", "pl_orbsmax": "semimajor_p",
        "pl_orbincl": "inc_p", "pl_rade": "radius_p", "pl_bmasse": "mass_p",
        "pl_insol": "flux_p", "pl_trandep": "observed_depth_percent",
        "pl_trandur": "observed_duration_hr", "st_rad": "radius_s", "st_mass": "mass_s",
        "st_teff": "teff_s", "st_lum": "st_lum_log10", "sy_dist": "distance_s",
        "sy_tmag": "tmag", "sy_gaiamag": "gaiamag", "sy_kepmag": "kepmag",
        "tic_id": "ticid", "TICID": "ticid", "ID": "ticid",
        "kepoi_name": "planet_name", "koi_period": "p_orb", "koi_prad": "radius_p",
        "koi_sma": "semimajor_p", "koi_incl": "inc_p", "koi_depth": "observed_depth_ppm",
        "koi_duration": "observed_duration_hr", "koi_srad": "radius_s", "koi_smass": "mass_s",
        "koi_steff": "teff_s", "koi_insol": "flux_p", "koi_kepmag": "kepmag",
        "Tmag": "tmag", "tess_mag": "tmag", "tessmag": "tmag",
        "gaia_g_mag": "gaiamag", "Gmag": "gaiamag", "luminosity_s": "l_sun",
        "temp_s": "teff_s", "insolation": "flux_p",
    }
    for src_col, dst_col in rename.items():
        if src_col in df.columns and dst_col not in df.columns:
            df = df.rename(columns={src_col: dst_col})

    if "st_lum_log10" in df.columns and "l_sun" not in df.columns:
        df["l_sun"] = 10 ** pd.to_numeric(df["st_lum_log10"], errors="coerce")

    if "observed_depth_percent" in df.columns and "observed_depth_ppm" not in df.columns:
        df["observed_depth_ppm"] = pd.to_numeric(df["observed_depth_percent"], errors="coerce") * 1e4

    for col in [
        "ra", "dec", "ticid", "tmag", "gaiamag", "kepmag", "radius_p", "mass_p", "flux_p",
        "radius_s", "mass_s", "teff_s", "l_sun", "distance_s", "p_orb", "semimajor_p", "inc_p",
        "observed_depth_ppm", "observed_duration_hr", "tess_dilution", "dilution",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = fill_semimajor_axis(df)
    df["dataset_source"] = {
        "pscomppars": "NASA_PSCompPars",
        "nasa": "NASA_PSCompPars",
        "koi": "NASA_KOI",
    }.get(source, "P-Pop_simulated")
    return df


def fill_semimajor_axis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "semimajor_p" not in df.columns:
        df["semimajor_p"] = np.nan
    if "p_orb" not in df.columns:
        return df
    missing = df["semimajor_p"].isna()
    if not missing.any():
        return df
    mstar = (
        pd.to_numeric(df["mass_s"], errors="coerce").fillna(1.0)
        if "mass_s" in df.columns else pd.Series(1.0, index=df.index)
    )
    df.loc[missing, "semimajor_p"] = semimajor_axis_from_period(
        pd.to_numeric(df["p_orb"], errors="coerce"), mstar
    )[missing]
    df["semimajor_p_source"] = np.where(missing, "estimated_from_period", "catalog")
    return df


def add_basic_columns(
    catalog: pd.DataFrame,
    *,
    apply_mdwarf_tmag_correction: bool,
    teff_grid,
    g_minus_t,
    mbol_minus_t,
) -> pd.DataFrame:
    catalog = catalog.copy()
    if "tmag" not in catalog.columns:
        catalog["tmag"] = np.nan
    if "tess_tmag_source" not in catalog.columns:
        catalog["tess_tmag_source"] = "missing"
    catalog = fill_tmag_proxy(
        catalog,
        apply_mdwarf_tmag_correction=apply_mdwarf_tmag_correction,
        teff_grid=teff_grid,
        g_minus_t=g_minus_t,
        mbol_minus_t=mbol_minus_t,
    )
    if "stype" not in catalog.columns:
        teff = catalog.get("teff_s", pd.Series(np.nan, index=catalog.index))
        catalog["stype"] = infer_stellar_type(teff)
    if "habitable" not in catalog.columns:
        flux = pd.to_numeric(catalog.get("flux_p", pd.Series(np.nan, index=catalog.index)), errors="coerce")
        catalog["habitable"] = (flux >= 0.25) & (flux <= 2.0)
    if "l_sun" in catalog.columns and "luminosity_s" not in catalog.columns:
        catalog["luminosity_s"] = catalog["l_sun"]
    if "teff_s" in catalog.columns and "temp_s" not in catalog.columns:
        catalog["temp_s"] = catalog["teff_s"]
    return catalog


def fill_tmag_proxy(
    catalog: pd.DataFrame,
    *,
    apply_mdwarf_tmag_correction: bool,
    teff_grid,
    g_minus_t,
    mbol_minus_t,
) -> pd.DataFrame:
    catalog = catalog.copy()
    tmag = pd.to_numeric(catalog.get("tmag", pd.Series(np.nan, index=catalog.index)), errors="coerce")
    src = pd.Series("tmag_catalog", index=catalog.index, dtype=object)
    src[tmag.isna()] = "missing"

    teff = pd.to_numeric(catalog["teff_s"], errors="coerce") if "teff_s" in catalog.columns else None

    for col, name in [("gaiamag", "gaiamag_proxy"), ("kepmag", "kepmag_proxy")]:
        if col not in catalog.columns:
            continue
        val = pd.to_numeric(catalog[col], errors="coerce")
        use = tmag.isna() & val.notna()
        if not use.any():
            continue

        if col == "gaiamag" and apply_mdwarf_tmag_correction and teff is not None:
            g_t = pd.Series(
                np.interp(teff.clip(2500, 8000).to_numpy(float), teff_grid, g_minus_t),
                index=catalog.index,
            )
            corrected = val - g_t
            use_with_teff = use & teff.notna()
            use_no_teff = use & teff.isna()
            tmag = tmag.copy()
            tmag.loc[use_with_teff] = corrected.loc[use_with_teff]
            src.loc[use_with_teff] = "gaiamag_teff_corrected_proxy"
            tmag.loc[use_no_teff] = val.loc[use_no_teff]
            src.loc[use_no_teff] = "gaiamag_proxy"
        else:
            tmag = tmag.copy()
            tmag.loc[use] = val.loc[use]
            src.loc[use] = name

    if {"l_sun", "distance_s"}.issubset(catalog.columns):
        l = pd.to_numeric(catalog["l_sun"], errors="coerce").clip(lower=1e-12)
        d = pd.to_numeric(catalog["distance_s"], errors="coerce").clip(lower=1e-12)
        mbol = pd.Series(apparent_bolometric_mag(l, d), index=catalog.index)

        if apply_mdwarf_tmag_correction and teff is not None:
            bc_t = pd.Series(
                np.interp(teff.clip(2500, 8000).to_numpy(float), teff_grid, mbol_minus_t),
                index=catalog.index,
            )
            mbol_corrected = mbol - bc_t
            use_with_teff = tmag.isna() & mbol_corrected.notna() & teff.notna()
            use_no_teff = tmag.isna() & mbol.notna() & teff.isna()
            tmag = tmag.copy()
            tmag.loc[use_with_teff] = mbol_corrected.loc[use_with_teff]
            src.loc[use_with_teff] = "mbol_teff_corrected_proxy"
            tmag.loc[use_no_teff] = mbol.loc[use_no_teff]
            src.loc[use_no_teff] = "mbol_proxy"
        else:
            use = tmag.isna() & mbol.notna()
            tmag = tmag.copy()
            tmag.loc[use] = mbol.loc[use]
            src.loc[use] = "mbol_proxy"

    catalog["tess_tmag"] = tmag
    catalog["tess_tmag_source"] = src
    return catalog


def validate_detection_columns(catalog: pd.DataFrame) -> None:
    required = ["radius_p", "radius_s", "p_orb", "semimajor_p"]
    missing = [c for c in required if c not in catalog.columns]
    if missing:
        raise ValueError(f"Missing required TESS detection columns: {missing}")


def prepare_catalog(
    data,
    source: str,
    *,
    apply_mdwarf_tmag_correction: bool,
    teff_grid,
    g_minus_t,
    mbol_minus_t,
    validate_for_detection: bool,
) -> tuple[pd.DataFrame, str]:
    catalog = as_dataframe(data)
    source = infer_source(catalog, source)
    catalog = standardize(catalog, source)
    catalog = add_basic_columns(
        catalog,
        apply_mdwarf_tmag_correction=apply_mdwarf_tmag_correction,
        teff_grid=teff_grid,
        g_minus_t=g_minus_t,
        mbol_minus_t=mbol_minus_t,
    )
    if validate_for_detection:
        validate_detection_columns(catalog)
    return catalog, source


def get_table_value(row, names, default=np.nan):
    for name in names:
        try:
            if name in row.colnames:
                return row[name]
        except Exception:
            pass
    return default


def enrich_from_tic(catalog: pd.DataFrame, mast_max_rows: int) -> pd.DataFrame:
    """Optional small-sample TIC enrichment with astroquery. Use cache externally for big runs."""
    catalog = catalog.copy()
    if not {"ra", "dec"}.issubset(catalog.columns):
        return catalog
    try:
        from astroquery.mast import Catalogs
        from astropy.coordinates import SkyCoord
        import astropy.units as u
    except Exception:
        catalog["tic_query_status"] = "astroquery_or_astropy_not_installed"
        return catalog

    cols = ["ticid", "tess_tmag", "radius_s", "mass_s", "teff_s"]
    for col in cols:
        if col not in catalog.columns:
            catalog[col] = np.nan
    catalog["tic_query_status"] = "not_queried"

    for idx, row in catalog.head(mast_max_rows).iterrows():
        if pd.isna(row.get("ra")) or pd.isna(row.get("dec")):
            continue
        try:
            coord = SkyCoord(float(row.ra), float(row.dec), unit="deg")
            tab = Catalogs.query_region(coord, radius=5 * u.arcsec, catalog="TIC")
            if len(tab) == 0:
                catalog.at[idx, "tic_query_status"] = "no_match"
                continue
            r = tab[0]
            catalog.at[idx, "ticid"] = get_table_value(r, ["ID", "TICID", "ticid"])
            for target, candidates in (
                ("tess_tmag", ["Tmag", "tmag"]),
                ("radius_s", ["rad", "radius"]),
                ("mass_s", ["mass"]),
                ("teff_s", ["Teff", "teff"]),
            ):
                current = row.get(target)
                replacement = get_table_value(r, candidates)
                catalog.at[idx, target] = (
                    replacement
                    if pd.isna(current) and not pd.isna(replacement) else current
                )
            catalog.at[idx, "tess_tmag_source"] = "TIC_MAST"
            catalog.at[idx, "tic_query_status"] = "matched"
        except Exception as exc:
            catalog.at[idx, "tic_query_status"] = f"failed:{type(exc).__name__}"
    return catalog


def set_visibility(catalog, rows, sectors, sector_text, source, observed_real, sector_days, dutycycle, span=None):
    catalog = catalog.copy()
    n_rows = len(catalog)
    pairs = pd.DataFrame({"row": np.asarray(rows, dtype=np.int64),
                          "sector": np.asarray(sectors, dtype=np.int64)})
    pairs["span"] = 1 if span is None else np.asarray(span, dtype=np.int64)
    n_sec = np.bincount(pairs["row"], weights=pairs["span"], minlength=n_rows).astype(np.int64)
    catalog["tess_n_sectors"] = n_sec
    catalog["tess_sectors"] = sector_text
    catalog["tess_sector_source"] = source
    catalog["tess_observed_real"] = observed_real
    catalog["tess_observed"] = n_sec > 0
    catalog["tess_observed_days"] = n_sec * sector_days * dutycycle
    return catalog, pairs


def expand_rows(rows, counts):
    counts = np.asarray(counts, dtype=np.int64)
    rep = np.repeat(np.asarray(rows, dtype=np.int64), counts)
    within = np.arange(len(rep)) - np.repeat(np.cumsum(counts) - counts, counts)
    return rep, within


def tag_cvz(catalog: pd.DataFrame, cvz_ecliptic_lat_deg: float) -> pd.DataFrame:
    catalog = catalog.copy()
    if not {"ra", "dec"}.issubset(catalog.columns):
        catalog["tess_ecliptic_lon"] = np.nan
        catalog["tess_ecliptic_lat"] = np.nan
        catalog["tess_in_cvz"] = False
        return catalog
    ra_rad = np.deg2rad(pd.to_numeric(catalog["ra"], errors="coerce").to_numpy(float))
    dec_rad = np.deg2rad(pd.to_numeric(catalog["dec"], errors="coerce").to_numpy(float))
    eps = np.deg2rad(OBLIQUITY_DEG)
    sin_elat = (np.sin(dec_rad) * np.cos(eps)
                - np.cos(dec_rad) * np.sin(eps) * np.sin(ra_rad))
    elat = np.rad2deg(np.arcsin(np.clip(sin_elat, -1.0, 1.0)))
    elon = np.rad2deg(np.arctan2(np.sin(ra_rad) * np.cos(eps) + np.tan(dec_rad) * np.sin(eps),
                                 np.cos(ra_rad))) % 360.0
    catalog["tess_ecliptic_lon"] = elon
    catalog["tess_ecliptic_lat"] = elat
    catalog["tess_in_cvz"] = np.abs(elat) > cvz_ecliptic_lat_deg
    return catalog


def add_default_visibility(catalog, *, default_n_sectors, cvz_n_sectors, cvz_ecliptic_lat_deg,
                           sector_days, dutycycle, source="fixed_n_sectors"):
    catalog = tag_cvz(catalog, cvz_ecliptic_lat_deg)
    n_sec = np.full(len(catalog), default_n_sectors, dtype=np.int64)
    if cvz_n_sectors is not None:
        n_sec[catalog["tess_in_cvz"].to_numpy(bool)] = cvz_n_sectors
    rows = np.flatnonzero(n_sec > 0)
    return set_visibility(
        catalog, rows, np.full(len(rows), -1), "", source, n_sec > 0,
        sector_days, dutycycle, span=n_sec[rows],
    )


def add_catalog_visibility(catalog, *, cvz_ecliptic_lat_deg, sector_days, dutycycle):
    catalog = tag_cvz(catalog, cvz_ecliptic_lat_deg)
    lists = [[int(s) for s in re.split(r"[;,\s]+", str(t)) if s.strip().isdigit()]
             for t in catalog["tess_sectors"].fillna("")]
    counts = [len(x) for x in lists]
    rows, _ = expand_rows(np.arange(len(lists)), counts)
    sectors = np.array([s for x in lists for s in x], dtype=np.int64)
    text = [";".join(map(str, x)) for x in lists]
    return set_visibility(catalog, rows, sectors, text, "catalog", np.array(counts) > 0,
                          sector_days, dutycycle)


def add_grid_visibility(catalog, *, sector_grid_path, condition_on_observed, cvz_ecliptic_lat_deg,
                        sector_days, dutycycle):
    catalog = tag_cvz(catalog, cvz_ecliptic_lat_deg)
    z = np.load(sector_grid_path)
    n_lon, n_sinlat = int(z["n_lon"]), int(z["n_sinlat"])
    covered = np.unpackbits(z["covered_bits"], axis=1)[:, :int(z["max_sector"])].astype(bool)

    elon = catalog["tess_ecliptic_lon"].to_numpy(float)
    elat = catalog["tess_ecliptic_lat"].to_numpy(float)
    ok = np.isfinite(elon) & np.isfinite(elat)
    i_lon = np.floor(np.nan_to_num(elon) / 360.0 * n_lon).astype(np.int64) % n_lon
    i_lat = np.floor((np.sin(np.deg2rad(np.nan_to_num(elat))) + 1.0) / 2.0 * n_sinlat).astype(np.int64)
    cell = np.where(ok, np.clip(i_lat, 0, n_sinlat - 1) * n_lon + i_lon, 0)

    cnt_cell = covered.sum(axis=1)
    indptr = np.r_[0, np.cumsum(cnt_cell)]
    cell_sectors = np.nonzero(covered)[1] + 1
    n_real = np.where(ok, cnt_cell[cell], 0)
    rows, within = expand_rows(np.arange(len(cell)), n_real)
    sectors = cell_sectors[indptr[cell[rows]] + within]

    used = np.unique(cell[n_real > 0])
    cell_text = {c: ";".join(map(str, cell_sectors[indptr[c]:indptr[c + 1]])) for c in used}
    text = np.array([cell_text.get(c, "") for c in cell], dtype=object)
    text[n_real == 0] = ""
    source = np.where(n_real > 0, "tess-point_sky_grid", "not_covered_by_tess")

    if condition_on_observed:
        cell_abslat = np.abs(np.rad2deg(np.arcsin(-1.0 + (np.arange(n_sinlat) + 0.5) * 2.0 / n_sinlat)))
        cell_band = np.repeat(np.minimum(cell_abslat // 10, 8).astype(int), n_lon)
        band_median = np.array([np.median(cnt_cell[(cell_band == b) & (cnt_cell > 0)]) for b in range(9)])
        row_band = np.minimum(np.abs(np.nan_to_num(elat)) // 10, 8).astype(int)
        n_fill = np.where(ok, band_median[row_band], np.median(cnt_cell[cnt_cell > 0]))
        n_fill = np.where(n_real == 0, np.maximum(np.round(n_fill), 1), 0).astype(np.int64)
        fill_rows, fill_within = expand_rows(np.arange(len(cell)), n_fill)
        rows = np.r_[rows, fill_rows]
        sectors = np.r_[sectors, -(fill_within + 1)]
        source = np.where(n_fill > 0, "assumed_if_observed", source)

    return set_visibility(catalog, rows, sectors, text, source, n_real > 0, sector_days, dutycycle)


def prepare_visibility(catalog: pd.DataFrame, *, use_catalog_sectors: bool, use_tesspoint: bool,
                       sector_grid_path: Path, condition_on_observed: bool,
                       default_n_sectors: int, cvz_n_sectors: int | None,
                       cvz_ecliptic_lat_deg: float, sector_days: float, dutycycle: float):
    if use_catalog_sectors and "tess_sectors" in catalog.columns:
        return add_catalog_visibility(
            catalog,
            cvz_ecliptic_lat_deg=cvz_ecliptic_lat_deg,
            sector_days=sector_days,
            dutycycle=dutycycle,
        )
    if use_tesspoint and Path(sector_grid_path).exists():
        return add_grid_visibility(
            catalog,
            sector_grid_path=sector_grid_path,
            condition_on_observed=condition_on_observed,
            cvz_ecliptic_lat_deg=cvz_ecliptic_lat_deg,
            sector_days=sector_days,
            dutycycle=dutycycle,
        )
    source = "fixed_n_sectors" if not use_tesspoint else "no_sector_grid_fixed_n_sectors"
    return add_default_visibility(
        catalog,
        default_n_sectors=default_n_sectors,
        cvz_n_sectors=cvz_n_sectors,
        cvz_ecliptic_lat_deg=cvz_ecliptic_lat_deg,
        sector_days=sector_days,
        dutycycle=dutycycle,
        source=source,
    )


def read_mast_csv_with_metadata(path: Path) -> pd.DataFrame:
    """Read MAST CDPP/TCE-style CSVs with metadata rows before the header."""
    path = Path(path)
    lines = path.read_text(errors="replace").splitlines()
    best_i, best_score = 0, -1
    for i, line in enumerate(lines[:120]):
        low = line.lower()
        n_fields = line.count(",") + 1
        if n_fields < 5:
            continue
        hits = sum(w in low for w in ["tic", "tmag", "rrmscdpp"])
        score = 10 * hits + n_fields
        if hits and score > best_score:
            best_i, best_score = i, score
            break
        if score > best_score:
            best_i, best_score = i, score
    return pd.read_csv(path, skiprows=best_i, low_memory=False)


def load_cdpp_tables(cdpp_dir: Path | None, cdpp_cols: dict) -> pd.DataFrame:
    if cdpp_dir is None or not Path(cdpp_dir).exists():
        return pd.DataFrame()
    frames = []
    for path in sorted(Path(cdpp_dir).glob("*.csv")):
        try:
            d = read_mast_csv_with_metadata(path)
        except Exception:
            continue
        d.columns = [c.lower().replace(" ", "") for c in d.columns]
        if "ticid" not in d.columns:
            continue
        if "sector" not in d.columns:
            m = re.search(r"[-_]s(\d{4})[-_]", path.name.lower())
            d["sector"] = int(m.group(1)) if m else pd.NA
        keep = ["ticid", "sector", "tmag"] + [c for c in cdpp_cols.values() if c in d.columns]
        d = d[keep].copy()
        d["ticid"] = pd.to_numeric(d["ticid"], errors="coerce").astype("Int64")
        d["sector"] = pd.to_numeric(d["sector"], errors="coerce").astype("Int64")
        if "tmag" in d.columns:
            d["tmag"] = pd.to_numeric(d["tmag"], errors="coerce")
        for c in cdpp_cols.values():
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def grid_cell_centers():
    lon = (np.arange(GRID_N_LON) + 0.5) * 360.0 / GRID_N_LON
    sinlat = -1.0 + (np.arange(GRID_N_SINLAT) + 0.5) * 2.0 / GRID_N_SINLAT
    lon2, sinlat2 = np.meshgrid(lon, sinlat)
    return lon2.ravel(), np.rad2deg(np.arcsin(sinlat2.ravel()))


def ecliptic_to_equatorial(lon_deg, lat_deg):
    lam, beta, eps = np.deg2rad(lon_deg), np.deg2rad(lat_deg), np.deg2rad(OBLIQUITY_DEG)
    dec = np.arcsin(np.sin(beta) * np.cos(eps) + np.cos(beta) * np.sin(eps) * np.sin(lam))
    ra = np.arctan2(np.sin(lam) * np.cos(eps) - np.tan(beta) * np.sin(eps), np.cos(lam))
    return np.mod(np.rad2deg(ra), 360.0), np.rad2deg(dec)


def build_grid(chunk=5000):
    import tess_stars2px
    from importlib.metadata import version

    lon, lat = grid_cell_centers()
    ra, dec = ecliptic_to_equatorial(lon, lat)
    covered = np.zeros((len(ra), MAX_SECTOR), dtype=bool)
    t0 = time.time()
    for start in range(0, len(ra), chunk):
        ids = np.arange(start, min(start + chunk, len(ra)))
        out = tess_stars2px.tess_stars2px_function_entry(ids, ra[ids], dec[ids])
        sid, sec = np.asarray(out[0]), np.asarray(out[3])
        keep = (sec >= 1) & (sec <= MAX_SECTOR)
        covered[sid[keep], sec[keep] - 1] = True
        print(f"  grid cells {ids[-1] + 1:,}/{len(ra):,}  ({time.time() - t0:.0f}s)", flush=True)

    TESS_REFERENCE_DATA_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        TESS_REFERENCE_DATA_DIR / "sector_grid.npz",
        covered_bits=np.packbits(covered, axis=1),
        n_lon=GRID_N_LON, n_sinlat=GRID_N_SINLAT, max_sector=MAX_SECTOR,
        obliquity_deg=OBLIQUITY_DEG, tess_point_version=version("tess-point"),
    )
    n = covered.sum(axis=1)
    print(f"Saved sector_grid.npz: {np.mean(n > 0):.1%} of sky covered through S{MAX_SECTOR}, "
          f"median {np.median(n[n > 0]):.0f} sectors where covered")


def build_noise(cdpp_cols: dict, cdpp_dir=TESS_DATA_DIR / "CDPP"):
    tab = load_cdpp_tables(Path(cdpp_dir), cdpp_cols)
    if tab.empty:
        raise FileNotFoundError(f"No SPOC CDPP CSVs in {cdpp_dir}")
    tab = tab[tab["sector"] <= MAX_SECTOR].dropna(subset=["tmag"])
    cols = list(cdpp_cols.values())
    tab["tmag_bin"] = np.round(tab["tmag"] / TMAG_BIN) * TMAG_BIN

    per_sector = tab.groupby(["sector", "tmag_bin"])[cols].median()
    per_sector["n_stars"] = tab.groupby(["sector", "tmag_bin"]).size()
    all_sectors = tab.groupby("tmag_bin")[cols].median()
    all_sectors["n_stars"] = tab.groupby("tmag_bin").size()
    all_sectors = pd.concat({0: all_sectors}, names=["sector"])

    out = pd.concat([all_sectors, per_sector]).reset_index()
    out = out[out["n_stars"] >= MIN_STARS_PER_BIN]
    out["sector"] = out["sector"].astype(int)
    TESS_REFERENCE_DATA_DIR.mkdir(exist_ok=True)
    out.round(3).to_csv(TESS_REFERENCE_DATA_DIR / "spoc_cdpp_tmag.csv", index=False)
    print(f"Saved spoc_cdpp_tmag.csv: {len(out):,} rows from {len(tab):,} target-sectors, "
          f"sectors {tab['sector'].min()}-{tab['sector'].max()}, "
          f"Tmag {out['tmag_bin'].min()}-{out['tmag_bin'].max()}")


def print_tmag_corrections(teff_grid):
    from science.catalogs import read_nasa_csv
    from tools.paths import PSCOMPPARS_CSV

    cols = ["st_teff", "st_lum", "sy_dist", "sy_tmag", "sy_gaiamag", "st_logg"]
    d = read_nasa_csv(PSCOMPPARS_CSV).drop_duplicates("hostname")[cols].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=["st_teff", "st_lum", "sy_dist", "sy_tmag"])
    d = d[(d["st_logg"] >= 4.0) | d["st_logg"].isna()]
    mbol = apparent_bolometric_mag(d["st_lum"], d["sy_dist"])
    edges = np.r_[2250, (teff_grid[1:] + teff_grid[:-1]) / 2, 9000]
    b = pd.cut(d["st_teff"], edges, labels=teff_grid)
    out = pd.DataFrame({"n": d.groupby(b, observed=False).size(),
                        "mbol_minus_t": (mbol - d["sy_tmag"]).groupby(b, observed=False).median(),
                        "g_minus_t": (d["sy_gaiamag"] - d["sy_tmag"]).groupby(b, observed=False).median()})
    print(f"{len(d):,} dwarf hosts\n{out.round(2).to_string()}")


"""
tess_data.py — concise TESS detector with tess-point + TIC/MAST + SPOC RMS CDPP support.

Main science use:
    P-Pop planets keep their generated mass/radius labels.
    TESSData only decides whether TESS would detect the transit.

Detection rule (hard threshold):
    detected = observed_by_tess AND transiting AND bright_enough
               AND enough_transits AND SNR >= snr_threshold

Detection probability weight (tess_p_detect):
    Always populated; used as weight for expected-detection analyses.
    For detection_model='threshold': 0.0 or 1.0 (same as hard cut).
    For detection_model='sigmoid':  logistic(sigmoid_steepness * (SNR - snr_threshold)).
    Used as sum(p_i) per bin — NOT as a Bernoulli draw — so results are fully reproducible.
    Rationale: even with exact synthetic parameters, a real telescope sees one noise
    realization per target; near-threshold planets are stochastically detected or missed.
    The logistic weight is the analytic stand-in for that photon-noise draw.
    All upstream gates (observed, transiting, bright, enough transits) zero out tess_p_detect
    before the SNR term even applies.

High-impact upgrades applied:
    1. CVZ tagging: ecliptic latitude computed; fallback n_sectors bumped for CVZ stars.
    2. FFI cadence: smooth CDPP fallback scaled by sqrt(cadence_min/2) for non-SPOC targets.
    3. Sigmoid detection: logistic probability weight (see above) for near-threshold planets.
    4. Impact parameter: model duration shortened by sqrt(1-b^2) for inclined orbits.
    5. M-dwarf Tmag proxy: Teff-based color correction for Gaia and mbol proxy magnitudes.

Noise priority:
    1. Official SPOC robust RMS CDPP tables, matched by TIC ID + sector + duration.
    2. Empirical median CDPP from those tables, binned by Tmag + duration.
    3. Smooth Tmag noise fallback (optionally scaled for FFI cadence).

Expected local CDPP files:
    cdpp_dir/*.csv with columns like:
        ticid, tmag, rrmscdpp00p5, rrmscdpp01p0, ..., rrmscdpp15p0
    Sector is read from a column named sector if present, otherwise from filenames
    containing patterns like s0001 or -s0001-s0001-.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import re

import numpy as np
import pandas as pd

try:
    from lifesim.core.data import Data
except Exception:  # lets this file import outside the lifesim environment
    Data = None


class TESSData:
    R_SUN_AU = 0.00465047
    R_EARTH_AU = 4.26352e-5
    R_SUN_REARTH = 109.076

    CDPP_DURATIONS_HR = np.array([
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.5,
        5.0, 6.0, 7.5, 9.0, 10.5, 12.5, 15.0,
    ])
    CDPP_COLS = {
        0.5: "rrmscdpp00p5", 1.0: "rrmscdpp01p0", 1.5: "rrmscdpp01p5",
        2.0: "rrmscdpp02p0", 2.5: "rrmscdpp02p5", 3.0: "rrmscdpp03p0",
        3.5: "rrmscdpp03p5", 4.5: "rrmscdpp04p5", 5.0: "rrmscdpp05p0",
        6.0: "rrmscdpp06p0", 7.5: "rrmscdpp07p5", 9.0: "rrmscdpp09p0",
        10.5: "rrmscdpp10p5", 12.5: "rrmscdpp12p5", 15.0: "rrmscdpp15p0",
    }

    # Empirical detection-efficiency calibration against the OFFICIAL SPOC pipeline
    # SNR (ExoFOP TOI "Planet SNR").  The idealized boxcar SNR (depth x sqrt(sum
    # N_i/CDPP_i^2)) runs hot: script 48 measures the toy/official ratio rising
    # toward the detection cut -- ~1.52 in the SNR 7.1-20 bins (median 1.41
    # overall) -- so the toy crossed SNR>=7.1 too readily and over-detected
    # marginal planets.  This factor is anchored at the threshold region
    # (~1/1.52) so the calibrated ratio is ~1.0 exactly where the SNR>=7.1 cut
    # is decided; it folds in the matched-filter-vs-boxcar shape loss and
    # limb-darkening reduction of the *effective* depth.  Re-derive with
    # scripts/Detector_Calibration/05_tess_calibration.py (requires the "TESS Mag"->tmag fix so
    # the smooth-CDPP fallback is not corrupted by a missing magnitude).
    SNR_OFFICIAL_CALIBRATION = 0.66

    # Teff grid for proxy Tmag color corrections (upgrade #5).
    # TEFF_GRID and G_MINUS_T_GRID give G - T color term (subtract from G to get T).
    # MBOL_MINUS_T_GRID gives mbol - T correction (subtract from mbol to get T).
    # Sources: TIC (Stassun+2019), Sullivan+2015, rough empirical matching.
    _TEFF_GRID = np.array([2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000], dtype=float)
    _G_MINUS_T  = np.array([2.80, 2.50, 1.80, 1.10, 0.80, 0.60, 0.45, 0.35, 0.20, 0.15], dtype=float)
    _MBOL_MINUS_T = np.array([2.00, 1.80, 1.40, 1.00, 0.70, 0.50, 0.40, 0.35, 0.20, 0.10], dtype=float)

    def __init__(
        self,
        data: Union[pd.DataFrame, object],
        source: str = "auto",
        *,
        sector_days: float = 27.4,
        dutycycle: float = 0.92,
        default_n_sectors: int = 1,
        min_transits: int = 2,
        snr_threshold: float = 7.1,
        # Multiplicative calibration of toy SNR to the official SPOC pipeline SNR.
        # Default = SNR_OFFICIAL_CALIBRATION (~1/1.52 at threshold). Pass 1.0 to recover the
        # raw, uncalibrated (optimistic) boxcar SNR, e.g. for the before/after
        # comparison in scripts/Detector_Calibration/05_tess_calibration.py.
        snr_calibration: Optional[float] = None,
        tmag_limit: float = 16.0,
        phase_mode: str = "random",  # random or expected
        random_seed: int = 42,
        use_tesspoint: bool = True,
        use_mast_tic: bool = False,
        mast_max_rows: int = 500,
        cdpp_dir: Optional[Union[str, Path]] = None,
        use_cdpp_tables: bool = True,
        smooth_noise_ref_ppm_1hr: float = 60.0,
        smooth_noise_floor_ppm_1hr: float = 30.0,
        # upgrade #3: apply impact-parameter b correction to modelled transit duration
        apply_b_to_duration: bool = True,
        # upgrade #4: Teff-based color correction for proxy Tmag (critical for M dwarfs)
        apply_mdwarf_tmag_correction: bool = True,
        # upgrade #1: CVZ tagging and fallback sector count for CVZ stars
        cvz_ecliptic_lat_deg: float = 78.0,
        cvz_n_sectors: int = 13,
        # upgrade #2: FFI cadence scaling for smooth CDPP fallback.
        # None = calibrated as-is (matches SPOC 2-min targets).
        # Set to 30.0 for primary-mission FFI stars or 10.0 for extended-mission FFI stars.
        ffi_cadence_min: Optional[float] = None,
        # upgrade #3: detection efficiency model.
        # 'threshold' = hard SNR step at snr_threshold (default; backward-compatible).
        # 'sigmoid'   = logistic probability weight used for expected-detection analysis.
        #   Use as: expected_count = tess_p_detect.sum()  (NOT as a Bernoulli draw)
        #   Calibration note: sigmoid_steepness should ideally be fit to TESS injection-recovery.
        detection_model: str = "threshold",
        sigmoid_steepness: float = 1.5,
        validate_for_detection: bool = True,
    ):
        self.catalog = self._as_dataframe(data)
        self.source = self._infer_source(source)

        self.sector_days = float(sector_days)
        self.dutycycle = float(dutycycle)
        self.default_n_sectors = int(default_n_sectors)
        self.min_transits = int(min_transits)
        self.snr_threshold = float(snr_threshold)
        self.snr_calibration = (
            float(self.SNR_OFFICIAL_CALIBRATION) if snr_calibration is None
            else float(snr_calibration)
        )
        self.tmag_limit = float(tmag_limit)
        self.phase_mode = phase_mode.lower().strip()
        self.rng = np.random.default_rng(random_seed)

        self.use_tesspoint = bool(use_tesspoint)
        self.use_mast_tic = bool(use_mast_tic)
        self.mast_max_rows = int(mast_max_rows)
        self.cdpp_dir = Path(cdpp_dir) if cdpp_dir is not None else None
        self.use_cdpp_tables = bool(use_cdpp_tables)
        self.smooth_noise_ref_ppm_1hr = float(smooth_noise_ref_ppm_1hr)
        self.smooth_noise_floor_ppm_1hr = float(smooth_noise_floor_ppm_1hr)

        # upgrade parameters
        self.apply_b_to_duration = bool(apply_b_to_duration)
        self.apply_mdwarf_tmag_correction = bool(apply_mdwarf_tmag_correction)
        self.cvz_ecliptic_lat_deg = float(cvz_ecliptic_lat_deg)
        self.cvz_n_sectors = int(cvz_n_sectors)
        self.ffi_cadence_min = float(ffi_cadence_min) if ffi_cadence_min is not None else None
        self.detection_model = detection_model.lower().strip()
        self.sigmoid_steepness = float(sigmoid_steepness)

        self.cdpp_table = self._load_cdpp_tables(self.cdpp_dir) if self.use_cdpp_tables else pd.DataFrame()
        self.catalog = self._standardize(self.catalog, self.source)
        self._add_basic_columns()

        if self.use_mast_tic:
            self._enrich_from_tic()
        if self.use_tesspoint:
            self._add_tesspoint_visibility()
        else:
            self._add_default_visibility()

        if validate_for_detection:
            self._validate()

    # ------------------------------------------------------------------
    # Setup and standardization
    # ------------------------------------------------------------------

    @staticmethod
    def _as_dataframe(data) -> pd.DataFrame:
        if Data is not None and isinstance(data, Data):
            return pd.DataFrame(data.catalog).copy()
        if isinstance(data, pd.DataFrame):
            return data.copy()
        return pd.DataFrame(data).copy()

    def _infer_source(self, source: str) -> str:
        source = str(source).lower().strip()
        if source != "auto":
            return source
        cols = set(self.catalog.columns)
        if {"pl_name", "pl_rade", "pl_orbper"}.issubset(cols):
            return "pscomppars"
        if {"kepoi_name", "koi_prad", "koi_period"}.issubset(cols):
            return "koi"
        return "ppop"

    @staticmethod
    def _first(df: pd.DataFrame, names, default=np.nan) -> pd.Series:
        for name in names:
            if name in df.columns:
                return df[name]
        return pd.Series(default, index=df.index)

    @staticmethod
    def _num(s) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    def _standardize(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        df = df.copy()
        rename = {
            # NASA PSCompPars
            "pl_name": "planet_name", "hostname": "host_name", "tran_flag": "tran_flag",
            "ra": "ra", "dec": "dec", "pl_orbper": "p_orb", "pl_orbsmax": "semimajor_p",
            "pl_orbincl": "inc_p", "pl_rade": "radius_p", "pl_bmasse": "mass_p",
            "pl_insol": "flux_p", "pl_trandep": "observed_depth_percent",
            "pl_trandur": "observed_duration_hr", "st_rad": "radius_s", "st_mass": "mass_s",
            "st_teff": "teff_s", "st_lum": "st_lum_log10", "sy_dist": "distance_s",
            "sy_tmag": "tmag", "sy_gaiamag": "gaiamag", "sy_kepmag": "kepmag",
            "tic_id": "ticid", "TICID": "ticid", "ID": "ticid",
            # KOI-ish
            "kepoi_name": "planet_name", "koi_period": "p_orb", "koi_prad": "radius_p",
            "koi_sma": "semimajor_p", "koi_incl": "inc_p", "koi_depth": "observed_depth_ppm",
            "koi_duration": "observed_duration_hr", "koi_srad": "radius_s", "koi_smass": "mass_s",
            "koi_steff": "teff_s", "koi_insol": "flux_p", "koi_kepmag": "kepmag",
            # P-Pop aliases
            "Tmag": "tmag", "tess_mag": "tmag", "tessmag": "tmag",
            "gaia_g_mag": "gaiamag", "Gmag": "gaiamag", "luminosity_s": "l_sun",
            "temp_s": "teff_s", "insolation": "flux_p",
        }
        df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns and v not in df.columns})

        if "st_lum_log10" in df.columns and "l_sun" not in df.columns:
            df["l_sun"] = 10 ** self._num(df["st_lum_log10"])

        if "observed_depth_percent" in df.columns and "observed_depth_ppm" not in df.columns:
            df["observed_depth_ppm"] = self._num(df["observed_depth_percent"]) * 1e4

        for col in [
            "ra", "dec", "ticid", "tmag", "gaiamag", "kepmag", "radius_p", "mass_p", "flux_p",
            "radius_s", "mass_s", "teff_s", "l_sun", "distance_s", "p_orb", "semimajor_p", "inc_p",
            "observed_depth_ppm", "observed_duration_hr", "tess_dilution", "dilution",
        ]:
            if col in df.columns:
                df[col] = self._num(df[col])

        if "semimajor_p" not in df.columns or df["semimajor_p"].isna().any():
            df = self._fill_semimajor_axis(df)

        df["dataset_source"] = {"pscomppars": "NASA_PSCompPars", "nasa": "NASA_PSCompPars", "koi": "NASA_KOI"}.get(source, "P-Pop_simulated")
        return df

    def _fill_semimajor_axis(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "semimajor_p" not in df.columns:
            df["semimajor_p"] = np.nan
        if "p_orb" not in df.columns:
            return df
        missing = df["semimajor_p"].isna()
        if not missing.any():
            return df
        p_yr = self._num(df["p_orb"]) / 365.25
        mstar = self._num(df["mass_s"]).fillna(1.0) if "mass_s" in df.columns else pd.Series(1.0, index=df.index)
        df.loc[missing, "semimajor_p"] = (mstar * p_yr ** 2) ** (1 / 3)
        df["semimajor_p_source"] = np.where(missing, "estimated_from_period", "catalog")
        return df

    def _add_basic_columns(self) -> None:
        if "tmag" not in self.catalog.columns:
            self.catalog["tmag"] = np.nan
        if "tess_tmag_source" not in self.catalog.columns:
            self.catalog["tess_tmag_source"] = "missing"
        self._fill_tmag_proxy()
        if "stype" not in self.catalog.columns:
            teff = self.catalog.get("teff_s", pd.Series(np.nan, index=self.catalog.index))
            self.catalog["stype"] = teff.apply(self._stellar_type)
        if "habitable" not in self.catalog.columns:
            flux = self._num(self.catalog.get("flux_p", pd.Series(np.nan, index=self.catalog.index)))
            self.catalog["habitable"] = (flux >= 0.25) & (flux <= 2.0)
        if "l_sun" in self.catalog.columns and "luminosity_s" not in self.catalog.columns:
            self.catalog["luminosity_s"] = self.catalog["l_sun"]
        if "teff_s" in self.catalog.columns and "temp_s" not in self.catalog.columns:
            self.catalog["temp_s"] = self.catalog["teff_s"]

    def _fill_tmag_proxy(self) -> None:
        """Populate tess_tmag from catalog value or proxy, with Teff-based color correction
        for Gaia G and mbol proxies (upgrade #5).

        TESS bandpass peaks at ~780 nm, redder than V or bolometric. M dwarfs emit strongly
        in this band, so raw Gaia G or mbol overestimates Tmag (makes the star look fainter
        than it really is to TESS). The correction uses a Teff → color-term lookup table
        derived from TIC (Stassun+2019) and Sullivan+2015.
        """
        tmag = self._num(self.catalog.get("tmag", pd.Series(np.nan, index=self.catalog.index)))
        src = pd.Series("tmag_catalog", index=self.catalog.index, dtype=object)
        src[tmag.isna()] = "missing"

        teff = self._num(self.catalog["teff_s"]) if "teff_s" in self.catalog.columns else None

        for col, name in [("gaiamag", "gaiamag_proxy"), ("kepmag", "kepmag_proxy")]:
            if col not in self.catalog.columns:
                continue
            val = self._num(self.catalog[col])
            use = tmag.isna() & val.notna()
            if not use.any():
                continue

            if col == "gaiamag" and self.apply_mdwarf_tmag_correction and teff is not None:
                # Subtract G - T color term so that corrected value ≈ Tmag
                g_t = pd.Series(
                    np.interp(teff.clip(2500, 8000).to_numpy(float),
                              self._TEFF_GRID, self._G_MINUS_T),
                    index=self.catalog.index,
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

        if {"l_sun", "distance_s"}.issubset(self.catalog.columns):
            l = self._num(self.catalog["l_sun"]).clip(lower=1e-12)
            d = self._num(self.catalog["distance_s"]).clip(lower=1e-12)
            mbol = 4.74 - 2.5 * np.log10(l) + 5 * np.log10(d / 10.0)

            if self.apply_mdwarf_tmag_correction and teff is not None:
                # Subtract bolometric→T correction (TESS band brighter than bol for cool stars)
                bc_t = pd.Series(
                    np.interp(teff.clip(2500, 8000).to_numpy(float),
                              self._TEFF_GRID, self._MBOL_MINUS_T),
                    index=self.catalog.index,
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

        self.catalog["tess_tmag"] = tmag
        self.catalog["tess_tmag_source"] = src

    @staticmethod
    def _stellar_type(teff) -> str:
        if pd.isna(teff): return "Unknown"
        teff = float(teff)
        if teff >= 7500: return "A"
        if teff >= 6000: return "F"
        if teff >= 5200: return "G"
        if teff >= 3700: return "K"
        if teff > 0: return "M"
        return "Unknown"

    def _validate(self) -> None:
        required = ["radius_p", "radius_s", "p_orb", "semimajor_p"]
        missing = [c for c in required if c not in self.catalog.columns]
        if missing:
            raise ValueError(f"Missing required TESS detection columns: {missing}")

    # ------------------------------------------------------------------
    # Optional real TESS metadata
    # ------------------------------------------------------------------

    def _enrich_from_tic(self) -> None:
        """Optional small-sample TIC enrichment with astroquery. Use cache externally for big runs."""
        if not {"ra", "dec"}.issubset(self.catalog.columns):
            return
        try:
            from astroquery.mast import Catalogs
            from astropy.coordinates import SkyCoord
            import astropy.units as u
        except Exception:
            self.catalog["tic_query_status"] = "astroquery_or_astropy_not_installed"
            return

        cols = ["ticid", "tess_tmag", "radius_s", "mass_s", "teff_s"]
        for col in cols:
            if col not in self.catalog.columns:
                self.catalog[col] = np.nan
        self.catalog["tic_query_status"] = "not_queried"

        for idx, row in self.catalog.head(self.mast_max_rows).iterrows():
            if pd.isna(row.get("ra")) or pd.isna(row.get("dec")):
                continue
            try:
                coord = SkyCoord(float(row.ra), float(row.dec), unit="deg")
                tab = Catalogs.query_region(coord, radius=5 * u.arcsec, catalog="TIC")
                if len(tab) == 0:
                    self.catalog.at[idx, "tic_query_status"] = "no_match"
                    continue
                r = tab[0]
                self.catalog.at[idx, "ticid"] = self._get_table_value(r, ["ID", "TICID", "ticid"])
                self.catalog.at[idx, "tess_tmag"] = self._coalesce(row.get("tess_tmag"), self._get_table_value(r, ["Tmag", "tmag"]))
                self.catalog.at[idx, "radius_s"] = self._coalesce(row.get("radius_s"), self._get_table_value(r, ["rad", "radius"]))
                self.catalog.at[idx, "mass_s"] = self._coalesce(row.get("mass_s"), self._get_table_value(r, ["mass"]))
                self.catalog.at[idx, "teff_s"] = self._coalesce(row.get("teff_s"), self._get_table_value(r, ["Teff", "teff"]))
                self.catalog.at[idx, "tess_tmag_source"] = "TIC_MAST"
                self.catalog.at[idx, "tic_query_status"] = "matched"
            except Exception as exc:
                self.catalog.at[idx, "tic_query_status"] = f"failed:{type(exc).__name__}"

    @staticmethod
    def _get_table_value(row, names, default=np.nan):
        for name in names:
            try:
                if name in row.colnames:
                    return row[name]
            except Exception:
                pass
        return default

    @staticmethod
    def _coalesce(a, b):
        return b if pd.isna(a) and not pd.isna(b) else a

    def _add_tesspoint_visibility(self) -> None:
        if not {"ra", "dec"}.issubset(self.catalog.columns):
            self._add_default_visibility("no_ra_dec_default")
            return
        try:
            from tess_stars2px import tess_stars2px_function_entry
        except Exception:
            self._add_default_visibility("tesspoint_not_installed_default")
            return

        ids = np.arange(len(self.catalog))
        ra = self._num(self.catalog["ra"]).to_numpy(float)
        dec = self._num(self.catalog["dec"]).to_numpy(float)
        ok = np.isfinite(ra) & np.isfinite(dec)
        sectors = [[] for _ in range(len(self.catalog))]
        cameras = [[] for _ in range(len(self.catalog))]
        ccds = [[] for _ in range(len(self.catalog))]
        cols = [[] for _ in range(len(self.catalog))]
        rows = [[] for _ in range(len(self.catalog))]

        try:
            out = tess_stars2px_function_entry(ids[ok], ra[ok], dec[ok])
            out_id, _elng, _elat, out_sec, out_cam, out_ccd, out_col, out_row = out[:8]
            for sid, sec, cam, ccd, col, row in zip(out_id, out_sec, out_cam, out_ccd, out_col, out_row):
                i = int(sid)
                sectors[i].append(int(sec)); cameras[i].append(int(cam)); ccds[i].append(int(ccd))
                cols[i].append(float(col)); rows[i].append(float(row))
            self.catalog["tess_sector_source"] = "tess-point"
        except Exception as exc:
            self._add_default_visibility(f"tesspoint_failed:{type(exc).__name__}")
            return

        self.catalog["tess_sectors"] = [self._join_ints(x) for x in sectors]
        self.catalog["tess_cameras"] = [self._join_ints(x) for x in cameras]
        self.catalog["tess_ccds"] = [self._join_ints(x) for x in ccds]
        self.catalog["tess_colpix"] = [self._join_floats(x) for x in cols]
        self.catalog["tess_rowpix"] = [self._join_floats(x) for x in rows]
        self.catalog["tess_n_sectors"] = [len(set(x)) for x in sectors]
        self.catalog["tess_observed"] = self.catalog["tess_n_sectors"] > 0
        self.catalog["tess_observed_days"] = self.catalog["tess_n_sectors"] * self.sector_days * self.dutycycle
        # upgrade #1: tag CVZ stars (tess-point handles actual sector counts, but tag is useful)
        self._tag_cvz()

    def _add_default_visibility(self, source="default") -> None:
        # upgrade #1: compute ecliptic latitude first so CVZ stars get more sectors
        self._tag_cvz()
        n_sec = pd.Series(self.default_n_sectors, index=self.catalog.index, dtype=int)
        if "tess_in_cvz" in self.catalog.columns:
            n_sec = n_sec.where(~self.catalog["tess_in_cvz"].astype(bool), self.cvz_n_sectors)
        self.catalog["tess_n_sectors"] = n_sec
        self.catalog["tess_sectors"] = [
            "" if n <= 0 else ";".join(map(str, range(1, n + 1)))
            for n in n_sec
        ]
        self.catalog["tess_sector_source"] = source
        self.catalog["tess_observed"] = n_sec > 0
        self.catalog["tess_observed_days"] = n_sec * self.sector_days * self.dutycycle

    def _tag_cvz(self) -> None:
        """Tag stars in TESS Continuous Viewing Zones (|ecliptic latitude| > cvz_ecliptic_lat_deg).

        Uses mean obliquity ε = 23.439°. CVZ stars near the ecliptic poles are observed
        every TESS cycle (~13 sectors/year), yielding much longer baselines and higher SNR.
        Stored in tess_ecliptic_lat (degrees) and tess_in_cvz (bool).
        """
        if not {"ra", "dec"}.issubset(self.catalog.columns):
            self.catalog["tess_ecliptic_lat"] = np.nan
            self.catalog["tess_in_cvz"] = False
            return
        ra_rad = np.deg2rad(self._num(self.catalog["ra"]).to_numpy(float))
        dec_rad = np.deg2rad(self._num(self.catalog["dec"]).to_numpy(float))
        eps = np.deg2rad(23.439)
        sin_elat = (np.sin(dec_rad) * np.cos(eps)
                    - np.cos(dec_rad) * np.sin(eps) * np.sin(ra_rad))
        elat = np.rad2deg(np.arcsin(np.clip(sin_elat, -1.0, 1.0)))
        self.catalog["tess_ecliptic_lat"] = elat
        self.catalog["tess_in_cvz"] = np.abs(elat) > self.cvz_ecliptic_lat_deg

    @staticmethod
    def _join_ints(values):
        return ";".join(str(int(v)) for v in values)

    @staticmethod
    def _join_floats(values):
        return ";".join(f"{float(v):.2f}" for v in values)

    # ------------------------------------------------------------------
    # CDPP tables and noise
    # ------------------------------------------------------------------

    @staticmethod
    def _read_mast_csv_with_metadata(path: Path) -> pd.DataFrame:
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

    def _load_cdpp_tables(self, cdpp_dir: Optional[Path]) -> pd.DataFrame:
        if cdpp_dir is None or not cdpp_dir.exists():
            return pd.DataFrame()
        frames = []
        for path in sorted(cdpp_dir.glob("*.csv")):
            try:
                d = self._read_mast_csv_with_metadata(path)
            except Exception:
                continue
            d.columns = [c.lower().replace(" ", "") for c in d.columns]
            if "ticid" not in d.columns:
                continue
            if "sector" not in d.columns:
                d["sector"] = self._sector_from_filename(path.name)
            keep = ["ticid", "sector", "tmag"] + [c for c in self.CDPP_COLS.values() if c in d.columns]
            d = d[keep].copy()
            d["ticid"] = self._num(d["ticid"]).astype("Int64")
            d["sector"] = self._num(d["sector"]).astype("Int64")
            if "tmag" in d.columns:
                d["tmag"] = self._num(d["tmag"])
            for c in self.CDPP_COLS.values():
                if c in d.columns:
                    d[c] = self._num(d[c])
            frames.append(d)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _sector_from_filename(name: str):
        # Match -s0001- or _s0001_ (not the year digits in 'tess2018...')
        m = re.search(r"[-_]s(\d{4})[-_]", name.lower())
        return int(m.group(1)) if m else pd.NA

    def _nearest_cdpp_col(self, duration_hr: float) -> tuple[float, str]:
        dur = float(np.clip(duration_hr, self.CDPP_DURATIONS_HR.min(), self.CDPP_DURATIONS_HR.max()))
        nearest = float(self.CDPP_DURATIONS_HR[np.argmin(np.abs(self.CDPP_DURATIONS_HR - dur))])
        return nearest, self.CDPP_COLS[nearest]

    def _cdpp_for_row_sector(self, ticid, sector, tmag, duration_hr):
        nearest, col = self._nearest_cdpp_col(duration_hr)
        if self.cdpp_table.empty or col not in self.cdpp_table.columns:
            return self._smooth_cdpp(duration_hr, tmag), nearest, "smooth_tmag_fallback_no_table"

        tic = pd.NA if pd.isna(ticid) else int(ticid)
        sec = pd.NA if pd.isna(sector) else int(sector)

        if not pd.isna(tic):
            m = self.cdpp_table["ticid"].eq(tic)
            if not pd.isna(sec) and "sector" in self.cdpp_table.columns:
                m = m & self.cdpp_table["sector"].eq(sec)
            val = self.cdpp_table.loc[m, col].dropna()
            if len(val):
                return float(val.median()), nearest, "official_spoc_cdpp_tic_sector"

            val = self.cdpp_table.loc[self.cdpp_table["ticid"].eq(tic), col].dropna()
            if len(val):
                return float(val.median()), nearest, "official_spoc_cdpp_tic_any_sector"

        if not pd.isna(tmag) and "tmag" in self.cdpp_table.columns:
            bin_half_width = 0.25
            m = self.cdpp_table["tmag"].between(float(tmag) - bin_half_width, float(tmag) + bin_half_width)
            if not pd.isna(sec) and "sector" in self.cdpp_table.columns:
                val = self.cdpp_table.loc[m & self.cdpp_table["sector"].eq(sec), col].dropna()
                if len(val) >= 10:
                    return float(val.median()), nearest, "empirical_cdpp_tmag_sector_bin"
            val = self.cdpp_table.loc[m, col].dropna()
            if len(val) >= 10:
                return float(val.median()), nearest, "empirical_cdpp_tmag_bin"

        return self._smooth_cdpp(duration_hr, tmag), nearest, "smooth_tmag_fallback"

    def _smooth_cdpp(self, duration_hr, tmag):
        """Smooth photon-noise CDPP fallback.

        Upgrade #2: if ffi_cadence_min is set, scale by sqrt(cadence_min / 2) to account
        for the coarser time sampling of FFI targets vs 2-min SPOC cadence. For 30-min primary
        FFI this factor is ~sqrt(15) ≈ 3.87; for 10-min extended FFI it is ~sqrt(5) ≈ 2.24.
        SPOC-matched rows (official_spoc_cdpp*) are already at 2-min resolution and are not
        passed through this function.
        """
        if pd.isna(tmag):
            return 5000.0
        noise_1hr = np.sqrt(
            self.smooth_noise_floor_ppm_1hr ** 2
            + self.smooth_noise_ref_ppm_1hr ** 2 * 10 ** (0.4 * (float(tmag) - 10.0))
        )
        cdpp = float(noise_1hr / np.sqrt(max(float(duration_hr), 0.1)))
        if self.ffi_cadence_min is not None:
            cdpp *= np.sqrt(self.ffi_cadence_min / 2.0)
        return cdpp

    # ------------------------------------------------------------------
    # Transit detection functions
    # ------------------------------------------------------------------

    def transiting(self) -> pd.Series:
        """True if the planet crosses the stellar disk; NASA transiting catalogs may use tran_flag."""
        if self.source in {"pscomppars", "nasa", "koi"} and "tran_flag" in self.catalog.columns:
            out = self._num(self.catalog["tran_flag"]).fillna(0).astype(int).eq(1)
            self.catalog["tess_transiting_source"] = "tran_flag"
            self.catalog["tess_impact_parameter_toy"] = np.nan
            return out
        if "inc_p" not in self.catalog.columns:
            raise ValueError("Need inc_p for geometric transits, or tran_flag for observed NASA transiting planets.")
        inc = self._num(self.catalog["inc_p"])
        inc_rad = inc if inc.max(skipna=True) <= 3.2 else np.deg2rad(inc)
        a = self._num(self.catalog["semimajor_p"])
        rs = self._num(self.catalog["radius_s"]) * self.R_SUN_AU
        rp = self._num(self.catalog["radius_p"]) * self.R_EARTH_AU
        b = a * np.abs(np.cos(inc_rad)) / rs
        self.catalog["tess_impact_parameter_toy"] = b
        self.catalog["tess_transiting_source"] = "inclination_geometry"
        return (b <= 1 + rp / rs).fillna(False)

    def depth_ppm(self) -> pd.Series:
        """Transit depth in ppm; use observed depth for NASA rows when present."""
        rp = self._num(self.catalog["radius_p"])
        rs = self._num(self.catalog["radius_s"]) * self.R_SUN_REARTH
        model = (rp / rs) ** 2 * 1e6
        observed = self._num(self.catalog.get("observed_depth_ppm", pd.Series(np.nan, index=self.catalog.index)))
        depth = observed.fillna(model)
        self.catalog["tess_transit_depth_source"] = np.where(observed.notna(), "observed", "model_Rp_Rstar")
        return depth

    def duration_hr(self) -> pd.Series:
        """Transit duration in hours; use observed duration when available, else circular-orbit model.

        Upgrade #3: when apply_b_to_duration=True and tess_impact_parameter_toy is available,
        the central-chord duration T0 = (P/π)(Rs/a) is multiplied by sqrt(1 - b^2) to account
        for the shorter chord length of inclined transits. This shortens duration for high-b
        planets, which reduces transit counts and SNR — particularly important near the detection
        limit. Only applied to modelled durations; observed durations from NASA are used as-is.
        """
        observed = self._num(self.catalog.get("observed_duration_hr", pd.Series(np.nan, index=self.catalog.index)))
        p = self._num(self.catalog["p_orb"])
        rs_au = self._num(self.catalog["radius_s"]) * self.R_SUN_AU
        a_au = self._num(self.catalog["semimajor_p"])
        model = (p * 24 / np.pi) * (rs_au / a_au)

        if self.apply_b_to_duration and "tess_impact_parameter_toy" in self.catalog.columns:
            b = self._num(self.catalog["tess_impact_parameter_toy"]).clip(0.0, 0.9999)
            b_factor = np.sqrt(1.0 - b ** 2).fillna(1.0)
            model = model * b_factor
            dur_source = np.where(observed.notna(), "observed", "model_with_b_correction")
        else:
            dur_source = np.where(observed.notna(), "observed", "model")

        dur = observed.fillna(model).replace([np.inf, -np.inf], np.nan).fillna(3.0).clip(0.25, 24.0)
        self.catalog["tess_transit_duration_source"] = dur_source
        return dur

    def transit_counts(self) -> list[dict[int, int]]:
        """Estimate how many transits land in each TESS sector."""
        counts_by_row = []
        p = self._num(self.catalog["p_orb"]).replace(0, np.nan)
        for idx, sectors_text in self.catalog["tess_sectors"].fillna("").items():
            sectors = [int(s) for s in str(sectors_text).split(";") if str(s).strip().isdigit()]
            if not sectors or pd.isna(p.loc[idx]):
                counts_by_row.append({})
                continue
            d = self.sector_days * self.dutycycle
            if self.phase_mode == "expected":
                n = max(0, int(np.floor(d / p.loc[idx])))
                counts_by_row.append({sec: n for sec in sectors})
            else:
                row_counts = {}
                for sec in sectors:
                    phase = self.rng.random() * p.loc[idx]
                    n = int(np.floor((d - phase) / p.loc[idx]) + 1) if d >= phase else 0
                    row_counts[sec] = max(n, 0)
                counts_by_row.append(row_counts)
        self.catalog["tess_n_transits"] = [sum(d.values()) for d in counts_by_row]
        self.catalog["tess_sector_transit_counts"] = [";".join(f"{k}:{v}" for k, v in d.items()) for d in counts_by_row]
        self.catalog["tess_enough_transits"] = self.catalog["tess_n_transits"] >= self.min_transits
        return counts_by_row

    def snr(self, counts_by_row: Optional[list[dict[int, int]]] = None) -> pd.Series:
        """TESS transit SNR using sector-duration CDPP: SNR = depth * sqrt(sum_i N_i / CDPP_i^2)."""
        if counts_by_row is None:
            counts_by_row = self.transit_counts()
        depth = self.depth_ppm()
        dur = self.duration_hr()
        tmag = self._num(self.catalog["tess_tmag"])
        tic = self._num(self.catalog.get("ticid", pd.Series(np.nan, index=self.catalog.index)))
        dilution = self._num(self.catalog.get("tess_dilution", self.catalog.get("dilution", pd.Series(1.0, index=self.catalog.index)))).fillna(1.0).clip(lower=1.0)

        snr_values, cdpp_values, cdpp_durs, noise_sources = [], [], [], []
        for i, idx in enumerate(self.catalog.index):
            row_sum = 0.0
            row_cdpps, row_durs, row_sources = [], [], []
            for sec, n in counts_by_row[i].items():
                if n <= 0:
                    continue
                c, c_dur, src = self._cdpp_for_row_sector(tic.loc[idx], sec, tmag.loc[idx], dur.loc[idx])
                if np.isfinite(c) and c > 0:
                    row_sum += n / c**2
                    row_cdpps.append(c); row_durs.append(c_dur); row_sources.append(src)
            snr_values.append(float(depth.loc[idx] / dilution.loc[idx] * np.sqrt(row_sum) * self.snr_calibration) if row_sum > 0 else 0.0)
            cdpp_values.append(float(np.median(row_cdpps)) if row_cdpps else np.nan)
            cdpp_durs.append(float(np.median(row_durs)) if row_durs else np.nan)
            noise_sources.append("+".join(sorted(set(row_sources))) if row_sources else "no_transits")

        self.catalog["tess_dilution"] = dilution
        self.catalog["tess_cdpp_ppm"] = cdpp_values
        self.catalog["tess_cdpp_duration_hr"] = cdpp_durs
        self.catalog["tess_noise_source"] = noise_sources
        self.catalog["tess_snr"] = snr_values
        self.catalog["tess_snr_threshold"] = self.snr_threshold
        return self.catalog["tess_snr"]

    def bright_enough(self) -> pd.Series:
        """TESS brightness gate; mainly prevents impossible/noisy missing-Tmag cases from passing."""
        bright = self._num(self.catalog["tess_tmag"]).le(self.tmag_limit).fillna(False)
        self.catalog["tess_tmag_limit"] = self.tmag_limit
        self.catalog["tess_star_bright_enough"] = bright
        return bright

    def classify_reasons(self) -> pd.Series:
        observed = self.catalog["tess_observed"].astype(bool)
        trans = self.catalog["tess_transiting_geometric"].astype(bool)
        bright = self.catalog["tess_star_bright_enough"].astype(bool)
        ntr = self._num(self.catalog["tess_n_transits"]).fillna(0)
        depth_pass = self.catalog["tess_depth_pass"].astype(bool)
        detected = self.catalog["tess_detected"].astype(bool)

        reason = np.full(len(self.catalog), "too_shallow_or_low_snr", dtype=object)
        reason[~observed.to_numpy()] = "not_observed_by_tess"
        reason[(observed & ~trans).to_numpy()] = "not_transiting"
        reason[(observed & trans & (ntr == 0)).to_numpy()] = "not_observed_in_time_window"
        reason[(observed & trans & (ntr == 1) & (self.min_transits > 1)).to_numpy()] = "single_transit_only"
        reason[(observed & trans & (ntr > 0) & (ntr < self.min_transits)).to_numpy()] = "too_few_transits"
        reason[(observed & trans & (ntr >= self.min_transits) & ~bright).to_numpy()] = "host_star_too_faint_or_noisy"
        reason[(observed & trans & (ntr >= self.min_transits) & bright & ~depth_pass).to_numpy()] = "too_shallow_or_low_snr"
        reason[detected.to_numpy()] = "detected"
        self.catalog["tess_reason_category"] = reason
        self.catalog["reason_category"] = reason
        self.catalog["miss_reason"] = reason
        return self.catalog["tess_reason_category"]

    def determine_detectable(self) -> pd.DataFrame:
        """Run the full TESS detector and add backward-compatible plotting columns."""
        self._validate()
        self.catalog["tess_transiting_geometric"] = self.transiting()
        self.catalog["transiting_geometric"] = self.catalog["tess_transiting_geometric"]
        # upgrade #3: duration_hr() reads tess_impact_parameter_toy set by transiting() above
        self.catalog["tess_transit_duration_hr"] = self.duration_hr()
        self.catalog["tess_transit_depth_ppm"] = self.depth_ppm()
        counts = self.transit_counts()
        self.catalog["tess_snr"] = self.snr(counts)
        self.catalog["tess_star_bright_enough"] = self.bright_enough()

        snr_col = self._num(self.catalog["tess_snr"])
        depth_pass = snr_col >= self.snr_threshold
        self.catalog["tess_depth_pass"] = depth_pass

        # upgrade #3: detection probability weight — NOT a Bernoulli draw.
        # tess_p_detect is 0/1 for threshold mode; smooth logistic for sigmoid mode.
        # Gated by all upstream conditions so non-observed / non-transiting planets
        # get weight 0 even if their SNR formula would give a non-zero value.
        # Usage: expected_detected_in_bin = tess_p_detect[mask].sum()
        all_gates = (
            self.catalog["tess_observed"].astype(bool)
            & self.catalog["tess_transiting_geometric"].astype(bool)
            & self.catalog["tess_star_bright_enough"].astype(bool)
            & self.catalog["tess_enough_transits"].astype(bool)
        )
        if self.detection_model == "sigmoid":
            p_snr = 1.0 / (1.0 + np.exp(-self.sigmoid_steepness * (snr_col - self.snr_threshold)))
        else:
            p_snr = depth_pass.astype(float)
        self.catalog["tess_p_detect"] = p_snr.where(all_gates, 0.0)

        detected = (
            self.catalog["tess_observed"].astype(bool)
            & self.catalog["tess_transiting_geometric"].astype(bool)
            & self.catalog["tess_star_bright_enough"].astype(bool)
            & self.catalog["tess_enough_transits"].astype(bool)
            & self.catalog["tess_depth_pass"].astype(bool)
        )
        self.catalog["tess_detected"] = detected

        # compatibility with old plotters
        self.catalog["detected"] = detected
        self.catalog["detected_best"] = detected
        self.catalog["detected_worst"] = detected
        self.catalog["transit_depth_ppm"] = self.catalog["tess_transit_depth_ppm"]
        self.catalog["tess_depth_pass_best"] = self.catalog["tess_depth_pass"]
        self.catalog["tess_depth_pass_worst"] = self.catalog["tess_depth_pass"]
        self.classify_reasons()
        return self.catalog

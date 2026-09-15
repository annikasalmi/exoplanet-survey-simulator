"""TESS toy transit detector for P-Pop and NASA tables. detected = observed by TESS, transiting,
bright enough, enough transits and SNR >= snr_threshold; tess_p_detect is a threshold or sigmoid
weight. By default every star is observed for 5 consecutive sectors, the same coverage for every star
as the Kepler detector gives its full mission; use_tesspoint=True takes the real pointings instead.
Noise per sector is SPOC's measured CDPP, by TIC when available, else the median of 2-min targets
at the same Tmag, else a smooth Tmag fallback. Reference files are built by build_reference_data.py.
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

    # Calibration against official SPOC SNR: on 1,551 SPOC multi-sector TOIs, given their searched
    # sectors and per-sector SPOC CDPP, the boxcar SNR runs 1.26x high (median over official SNR
    # 10-100); after this factor model/official is 0.92-1.03 in every SNR bin.
    # Re-derive with plotting/scripts/calibration/tess_calibration.py.
    SNR_OFFICIAL_CALIBRATION = 0.80

    DATA_DIR = Path(__file__).resolve().parent / "data"

    # Proxy-Tmag colour terms vs Teff: G - T and mbol - T (subtract from G or mbol to get T).
    # Medians over 4,078 dwarf planet hosts (log g >= 4) in NASA PSCompPars, whose sy_tmag is TIC-8
    # (Stassun et al. 2019); the 2500 K bin has only 4 stars. Re-derive with
    # `python telescopes/tess/build_reference_data.py tmag`.
    _TEFF_GRID = np.array([2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000], dtype=float)
    _G_MINUS_T  = np.array([1.79, 1.37, 1.16, 0.94, 0.71, 0.60, 0.50, 0.43, 0.32, 0.12], dtype=float)
    _MBOL_MINUS_T = np.array([-0.30, 0.19, 0.35, 0.42, 0.43, 0.46, 0.44, 0.39, 0.29, 0.13], dtype=float)

    # Priority order of noise sources; each row reports the best one any of its sectors used.
    NOISE_SOURCES = ["official_spoc_cdpp_tic_sector", "official_spoc_cdpp_tic_any_sector",
                     "spoc_cdpp_tmag_sector_bin", "spoc_cdpp_tmag_all_sectors", "smooth_tmag_fallback"]
    NOISE_TMAG_BIN = 0.5

    def __init__(
        self,
        data: Union[pd.DataFrame, object],
        source: str = "auto",
        *,
        sector_days: float = 27.4,
        dutycycle: float = 0.92,
        # Coverage when use_tesspoint is False: every star observed for this many consecutive sectors in
        # one continuous window, as the Kepler detector gives every star its full mission.
        default_n_sectors: int = 5,
        min_transits: int = 2,
        snr_threshold: float = 7.1,
        # Multiplicative calibration of toy SNR to the official SPOC pipeline SNR.
        # Default = SNR_OFFICIAL_CALIBRATION (1/1.26). Pass 1.0 to recover the raw, uncalibrated
        # (optimistic) boxcar SNR, as plotting/scripts/calibration/tess_calibration.py does.
        snr_calibration: Optional[float] = None,
        tmag_limit: float = 16.0,
        phase_mode: str = "random",  # random or expected
        random_seed: int = 42,
        # True: sectors from the real pointings in data/sector_grid.npz (tess-point on a sky grid).
        # False: every star gets default_n_sectors (cvz_n_sectors in the CVZ, if set).
        use_tesspoint: bool = False,
        # True answers "if TESS had looked, would it have found it?": stars the real pointings never
        # covered get the median sector count of covered sky at the same |ecliptic latitude| and
        # stay tess_observed. tess_observed_real records whether TESS really covered the star.
        condition_on_observed: bool = True,
        # Take sectors from an input tess_sectors column ("1;2;5") instead, e.g. for TOIs.
        use_catalog_sectors: bool = False,
        sector_grid_path: Optional[Union[str, Path]] = None,
        noise_table_path: Optional[Union[str, Path]] = None,
        use_mast_tic: bool = False,
        mast_max_rows: int = 500,
        # Per-TIC SPOC CDPP CSVs from MAST (results/catalogs/tess/CDPP); only rows with a ticid use them.
        cdpp_dir: Optional[Union[str, Path]] = None,
        use_cdpp_tables: bool = True,
        # Last-resort noise when no SPOC table covers a row: 1-hr noise ~ ticgen's pre-launch model
        # (Sullivan et al. 2015 with its 60 ppm/hr systematic floor), within 10% for Tmag 6-12.
        smooth_noise_ref_ppm_1hr: float = 200.0,
        smooth_noise_floor_ppm_1hr: float = 60.0,
        # upgrade #3: apply impact-parameter b correction to modelled transit duration
        apply_b_to_duration: bool = True,
        # upgrade #4: Teff-based color correction for proxy Tmag (critical for M dwarfs)
        apply_mdwarf_tmag_correction: bool = True,
        # CVZ tagging, and an optional larger fixed sector count for CVZ stars (None = no special case)
        cvz_ecliptic_lat_deg: float = 78.0,
        cvz_n_sectors: Optional[int] = None,
        # upgrade #2: FFI cadence scaling for smooth CDPP fallback.
        # None = calibrated as-is (matches SPOC 2-min targets).
        # Set to 30.0 for primary-mission FFI stars or 10.0 for extended-mission FFI stars.
        ffi_cadence_min: Optional[float] = None,
        # detection_model: 'threshold' = hard SNR cut (default); 'sigmoid' = logistic weight for
        # expected counts (tess_p_detect.sum(), not a Bernoulli draw). Steepness should ideally
        # be fit to TESS injection-recovery.
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
        self.condition_on_observed = bool(condition_on_observed)
        self.use_catalog_sectors = bool(use_catalog_sectors)
        self.sector_grid_path = Path(sector_grid_path) if sector_grid_path else self.DATA_DIR / "sector_grid.npz"
        self.noise_table_path = Path(noise_table_path) if noise_table_path else self.DATA_DIR / "spoc_cdpp_tmag.csv"
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
        self.cvz_n_sectors = int(cvz_n_sectors) if cvz_n_sectors is not None else None
        self.ffi_cadence_min = float(ffi_cadence_min) if ffi_cadence_min is not None else None
        self.detection_model = detection_model.lower().strip()
        self.sigmoid_steepness = float(sigmoid_steepness)

        self.cdpp_table = self._load_cdpp_tables(self.cdpp_dir) if self.use_cdpp_tables else pd.DataFrame()
        self.noise_table = pd.read_csv(self.noise_table_path) if self.noise_table_path.exists() else pd.DataFrame()
        self.catalog = self._standardize(self.catalog, self.source)
        self._add_basic_columns()

        if self.use_mast_tic:
            self._enrich_from_tic()
        if self.use_catalog_sectors and "tess_sectors" in self.catalog.columns:
            self._add_catalog_visibility()
        elif self.use_tesspoint and self.sector_grid_path.exists():
            self._add_grid_visibility()
        else:
            self._add_default_visibility("fixed_n_sectors" if not self.use_tesspoint else "no_sector_grid_fixed_n_sectors")

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
        # Apply one at a time: several source names can map to the same target
        # (st_teff and temp_s both mean teff_s), and a single comprehension
        # evaluates its guard against the original columns, so both would pass
        # and produce two columns with one name.
        for src_col, dst_col in rename.items():
            if src_col in df.columns and dst_col not in df.columns:
                df = df.rename(columns={src_col: dst_col})

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
        """Set tess_tmag from the catalog or a proxy. Gaia G and mbol proxies get a Teff-based color
        correction (TIC, Stassun+2019; Sullivan+2015), since M dwarfs are brighter in the TESS band.
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

    # Visibility produces self._pairs, one row per (catalog row, sector) the star is observed in.
    # Sectors < 0 are assumed ones (fixed count, or filled for stars the pointings missed); their
    # noise comes from the all-sector SPOC medians.

    def _set_visibility(self, rows, sectors, sector_text, source, observed_real, span=None) -> None:
        n_rows = len(self.catalog)
        self._pairs = pd.DataFrame({"row": np.asarray(rows, dtype=np.int64),
                                    "sector": np.asarray(sectors, dtype=np.int64)})
        # span = sectors one pair stands for: 1 for a real sector, n for a continuous n-sector window.
        self._pairs["span"] = 1 if span is None else np.asarray(span, dtype=np.int64)
        n_sec = np.bincount(self._pairs["row"], weights=self._pairs["span"], minlength=n_rows).astype(np.int64)
        self.catalog["tess_n_sectors"] = n_sec
        self.catalog["tess_sectors"] = sector_text
        self.catalog["tess_sector_source"] = source
        self.catalog["tess_observed_real"] = observed_real
        self.catalog["tess_observed"] = n_sec > 0
        self.catalog["tess_observed_days"] = n_sec * self.sector_days * self.dutycycle

    @staticmethod
    def _expand(rows, counts):
        """Repeat each row counts[i] times; also return 0..counts[i]-1 within each row."""
        counts = np.asarray(counts, dtype=np.int64)
        rep = np.repeat(np.asarray(rows, dtype=np.int64), counts)
        within = np.arange(len(rep)) - np.repeat(np.cumsum(counts) - counts, counts)
        return rep, within

    def _add_default_visibility(self, source="fixed_n_sectors") -> None:
        self._tag_cvz()
        n_sec = np.full(len(self.catalog), self.default_n_sectors, dtype=np.int64)
        if self.cvz_n_sectors is not None:
            n_sec[self.catalog["tess_in_cvz"].to_numpy(bool)] = self.cvz_n_sectors
        rows = np.flatnonzero(n_sec > 0)
        self._set_visibility(rows, np.full(len(rows), -1), "", source, n_sec > 0, span=n_sec[rows])

    def _add_catalog_visibility(self) -> None:
        self._tag_cvz()
        lists = [[int(s) for s in re.split(r"[;,\s]+", str(t)) if s.strip().isdigit()]
                 for t in self.catalog["tess_sectors"].fillna("")]
        counts = [len(x) for x in lists]
        rows, _ = self._expand(np.arange(len(lists)), counts)
        sectors = np.array([s for x in lists for s in x], dtype=np.int64)
        text = [";".join(map(str, x)) for x in lists]
        self._set_visibility(rows, sectors, text, "catalog", np.array(counts) > 0)

    def _add_grid_visibility(self) -> None:
        """Real sectors per star from the tess-point sky grid (see build_reference_data.py)."""
        z = np.load(self.sector_grid_path)
        n_lon, n_sinlat = int(z["n_lon"]), int(z["n_sinlat"])
        covered = np.unpackbits(z["covered_bits"], axis=1)[:, :int(z["max_sector"])].astype(bool)

        self._tag_cvz()
        elon = self.catalog["tess_ecliptic_lon"].to_numpy(float)
        elat = self.catalog["tess_ecliptic_lat"].to_numpy(float)
        ok = np.isfinite(elon) & np.isfinite(elat)
        i_lon = np.floor(np.nan_to_num(elon) / 360.0 * n_lon).astype(np.int64) % n_lon
        i_lat = np.floor((np.sin(np.deg2rad(np.nan_to_num(elat))) + 1.0) / 2.0 * n_sinlat).astype(np.int64)
        cell = np.where(ok, np.clip(i_lat, 0, n_sinlat - 1) * n_lon + i_lon, 0)

        # CSR layout: sectors of cell c are cell_sectors[indptr[c]:indptr[c + 1]].
        cnt_cell = covered.sum(axis=1)
        indptr = np.r_[0, np.cumsum(cnt_cell)]
        cell_sectors = np.nonzero(covered)[1] + 1
        n_real = np.where(ok, cnt_cell[cell], 0)
        rows, within = self._expand(np.arange(len(cell)), n_real)
        sectors = cell_sectors[indptr[cell[rows]] + within]

        used = np.unique(cell[n_real > 0])
        cell_text = {c: ";".join(map(str, cell_sectors[indptr[c]:indptr[c + 1]])) for c in used}
        text = np.array([cell_text.get(c, "") for c in cell], dtype=object)
        text[n_real == 0] = ""
        source = np.where(n_real > 0, "tess-point_sky_grid", "not_covered_by_tess")

        if self.condition_on_observed:
            # Median sector count of covered sky in 10-deg |ecliptic latitude| bands.
            cell_abslat = np.abs(np.rad2deg(np.arcsin(-1.0 + (np.arange(n_sinlat) + 0.5) * 2.0 / n_sinlat)))
            cell_band = np.repeat(np.minimum(cell_abslat // 10, 8).astype(int), n_lon)
            band_median = np.array([np.median(cnt_cell[(cell_band == b) & (cnt_cell > 0)]) for b in range(9)])
            row_band = np.minimum(np.abs(np.nan_to_num(elat)) // 10, 8).astype(int)
            n_fill = np.where(ok, band_median[row_band], np.median(cnt_cell[cnt_cell > 0]))
            n_fill = np.where(n_real == 0, np.maximum(np.round(n_fill), 1), 0).astype(np.int64)
            fill_rows, fill_within = self._expand(np.arange(len(cell)), n_fill)
            rows = np.r_[rows, fill_rows]
            sectors = np.r_[sectors, -(fill_within + 1)]
            source = np.where(n_fill > 0, "assumed_if_observed", source)

        self._set_visibility(rows, sectors, text, source, n_real > 0)

    def _tag_cvz(self) -> None:
        """Ecliptic coordinates (tess_ecliptic_lon/lat, deg) and tess_in_cvz: |ecliptic latitude|
        > cvz_ecliptic_lat_deg, the continuous viewing zones (~13 sectors a year).
        """
        if not {"ra", "dec"}.issubset(self.catalog.columns):
            self.catalog["tess_ecliptic_lon"] = np.nan
            self.catalog["tess_ecliptic_lat"] = np.nan
            self.catalog["tess_in_cvz"] = False
            return
        ra_rad = np.deg2rad(self._num(self.catalog["ra"]).to_numpy(float))
        dec_rad = np.deg2rad(self._num(self.catalog["dec"]).to_numpy(float))
        eps = np.deg2rad(23.439)
        sin_elat = (np.sin(dec_rad) * np.cos(eps)
                    - np.cos(dec_rad) * np.sin(eps) * np.sin(ra_rad))
        elat = np.rad2deg(np.arcsin(np.clip(sin_elat, -1.0, 1.0)))
        elon = np.rad2deg(np.arctan2(np.sin(ra_rad) * np.cos(eps) + np.tan(dec_rad) * np.sin(eps),
                                     np.cos(ra_rad))) % 360.0
        self.catalog["tess_ecliptic_lon"] = elon
        self.catalog["tess_ecliptic_lat"] = elat
        self.catalog["tess_in_cvz"] = np.abs(elat) > self.cvz_ecliptic_lat_deg

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

    def _nearest_cdpp_index(self, duration_hr) -> np.ndarray:
        dur = np.clip(np.asarray(duration_hr, dtype=float), self.CDPP_DURATIONS_HR.min(), self.CDPP_DURATIONS_HR.max())
        return np.abs(dur[:, None] - self.CDPP_DURATIONS_HR[None, :]).argmin(axis=1)

    def _pair_cdpp(self, rows, sectors, col_idx, tmag, ticid):
        """CDPP (ppm) and noise-source rank for each observed (row, sector) pair, best source first:
        SPOC CDPP for this TIC and sector, this TIC in any sector, the median of 2-min targets at the
        same Tmag in this sector, the same across all sectors, then the smooth fallback.
        """
        cols = list(self.CDPP_COLS.values())
        ci = col_idx[rows]
        t = tmag[rows]
        n = len(rows)
        cdpp = np.full(n, np.nan)
        rank = np.full(n, len(self.NOISE_SOURCES) - 1, dtype=np.int64)

        def take(values, r):
            nonlocal cdpp
            v = values[np.arange(n), ci]
            fill = np.isnan(cdpp) & np.isfinite(v) & (v > 0)
            cdpp[fill] = v[fill]
            rank[fill] = r

        if not self.cdpp_table.empty and np.isfinite(ticid).any():
            raw = self.cdpp_table.dropna(subset=["ticid"])
            by_sector = raw.groupby(["ticid", "sector"])[cols].median()
            tic = pd.array(np.where(np.isfinite(ticid[rows]), ticid[rows], -1), dtype="Int64")
            take(by_sector.reindex(pd.MultiIndex.from_arrays([tic, pd.array(sectors, dtype="Int64")])).to_numpy(float), 0)
            take(raw.groupby("ticid")[cols].median().reindex(tic).to_numpy(float), 1)

        if not self.noise_table.empty:
            nt = self.noise_table
            t_bin = np.round(np.nan_to_num(t, nan=-99.0) / self.NOISE_TMAG_BIN) * self.NOISE_TMAG_BIN
            per_sector = nt[nt["sector"] > 0].set_index(["sector", "tmag_bin"])[cols]
            take(per_sector.reindex(pd.MultiIndex.from_arrays([sectors, t_bin])).to_numpy(float), 2)
            # All-sector medians, interpolated in Tmag and held flat past the table's ends.
            allsec = nt[nt["sector"] == 0].sort_values("tmag_bin")
            interp = np.column_stack([np.interp(t, allsec["tmag_bin"], allsec[c]) for c in cols])
            take(interp, 3)

        missing = np.isnan(cdpp)
        cdpp[missing] = self._smooth_cdpp(self.CDPP_DURATIONS_HR[ci[missing]], t[missing])
        return cdpp, rank

    def _smooth_cdpp(self, duration_hr, tmag):
        """Smooth photon-plus-floor CDPP fallback. With ffi_cadence_min set, scales by
        sqrt(cadence_min / 2) for FFI targets (~3.87 at 30 min, ~2.24 at 10 min).
        """
        tmag = np.asarray(tmag, dtype=float)
        noise_1hr = np.sqrt(self.smooth_noise_floor_ppm_1hr ** 2
                            + self.smooth_noise_ref_ppm_1hr ** 2 * 10 ** (0.4 * (tmag - 10.0)))
        cdpp = noise_1hr / np.sqrt(np.maximum(np.asarray(duration_hr, dtype=float), 0.1))
        if self.ffi_cadence_min is not None:
            cdpp = cdpp * np.sqrt(self.ffi_cadence_min / 2.0)
        return np.where(np.isfinite(tmag), cdpp, 5000.0)

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
        """Transit duration (h): the observed value when available, else a circular-orbit model,
        shortened by sqrt(1 - b^2) when apply_b_to_duration is set.
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

    def transit_counts(self) -> pd.DataFrame:
        """Transits in each observed window, one row per (catalog row, sector or continuous span).
        A window lasts span * sector_days * dutycycle days, like the Kepler detector's
        dataspan * dutycycle. phase_mode 'random' draws one phase per window; 'expected' uses the
        mean count d / P, which is fractional, so a planet with one transit every other sector counts.
        """
        pairs = self._pairs.copy()
        p = self._num(self.catalog["p_orb"]).to_numpy(float)[pairs["row"].to_numpy()]
        d = pairs["span"].to_numpy() * self.sector_days * self.dutycycle
        good = np.isfinite(p) & (p > 0)
        p = np.where(good, p, 1.0)
        if self.phase_mode == "expected":
            n = d / p
        else:
            phase = self.rng.random(len(pairs)) * p
            n = np.where(d >= phase, np.floor((d - phase) / p) + 1, 0)
        pairs["n"] = np.where(good, n, 0.0)
        n_total = np.bincount(pairs["row"], weights=pairs["n"], minlength=len(self.catalog))
        if self.phase_mode != "expected":
            n_total = n_total.astype(np.int64)
        self.catalog["tess_n_transits"] = n_total
        self.catalog["tess_enough_transits"] = n_total >= self.min_transits
        return pairs

    def snr(self, pairs: Optional[pd.DataFrame] = None) -> pd.Series:
        """TESS transit SNR over all observed sectors: SNR = depth * sqrt(sum_i N_i / CDPP_i^2).
        tess_cdpp_ppm is the effective per-transit CDPP, sqrt(sum N_i / sum(N_i / CDPP_i^2)).
        """
        if pairs is None:
            pairs = self.transit_counts()
        n_rows = len(self.catalog)
        depth = self.depth_ppm().to_numpy(float)
        dur = self.duration_hr().to_numpy(float)
        tmag = self._num(self.catalog["tess_tmag"]).to_numpy(float)
        tic = self._num(self.catalog.get("ticid", pd.Series(np.nan, index=self.catalog.index))).to_numpy(float)
        dilution = self._num(self.catalog.get("tess_dilution", self.catalog.get("dilution", pd.Series(1.0, index=self.catalog.index)))).fillna(1.0).clip(lower=1.0)
        col_idx = self._nearest_cdpp_index(dur)

        obs = pairs[pairs["n"] > 0]
        rows = obs["row"].to_numpy()
        cdpp, rank = self._pair_cdpp(rows, obs["sector"].to_numpy(), col_idx, tmag, tic)
        inv_var = np.bincount(rows, weights=obs["n"].to_numpy() / cdpp ** 2, minlength=n_rows)
        n_used = np.bincount(rows, weights=obs["n"].to_numpy(), minlength=n_rows)
        best = np.full(n_rows, len(self.NOISE_SOURCES), dtype=np.int64)
        np.minimum.at(best, rows, rank)

        snr = np.where(inv_var > 0, depth / dilution.to_numpy() * np.sqrt(inv_var) * self.snr_calibration, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.catalog["tess_cdpp_ppm"] = np.where(inv_var > 0, np.sqrt(n_used / inv_var), np.nan)
        self.catalog["tess_dilution"] = dilution
        self.catalog["tess_cdpp_duration_hr"] = np.where(inv_var > 0, self.CDPP_DURATIONS_HR[col_idx], np.nan)
        self.catalog["tess_noise_source"] = np.array(self.NOISE_SOURCES + ["no_transits"], dtype=object)[best]
        self.catalog["tess_snr"] = snr
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
        return self.catalog["tess_reason_category"]

    def determine_detectable(self) -> pd.DataFrame:
        """Run the full TESS detector and add the shared detected/detected_best/detected_worst columns."""
        self._validate()
        self.catalog["tess_transiting_geometric"] = self.transiting()
        # upgrade #3: duration_hr() reads tess_impact_parameter_toy set by transiting() above
        self.catalog["tess_transit_duration_hr"] = self.duration_hr()
        self.catalog["tess_transit_depth_ppm"] = self.depth_ppm()
        counts = self.transit_counts()
        self.catalog["tess_snr"] = self.snr(counts)
        self.catalog["tess_star_bright_enough"] = self.bright_enough()

        snr_col = self._num(self.catalog["tess_snr"])
        depth_pass = snr_col >= self.snr_threshold
        self.catalog["tess_depth_pass"] = depth_pass

        # Detection probability weight, not a Bernoulli draw: 0/1 for threshold, logistic for sigmoid,
        # zero unless observed and transiting. Use tess_p_detect[mask].sum().
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

        # TESS has one scenario; the shared plotters read best/worst (real for HWO and LIFEsim).
        self.catalog["detected"] = detected
        self.catalog["detected_best"] = detected
        self.catalog["detected_worst"] = detected
        self.classify_reasons()
        return self.catalog

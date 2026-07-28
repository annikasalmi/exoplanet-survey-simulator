"""
kepler_data.py — dual P-Pop + NASA PSCompPars/KOI Kepler toy detector

Safe replacement goal:
1. P-Pop input still works:
       KeplerData(PPopObj.catalog).determine_detectable()

2. NASA PSCompPars input works:
       KeplerData(df, source='pscomppars', ...).determine_detectable()

3. Same simple detector remains:
       detected = transiting and bright_enough_kepler and depth_good

4. NASA-specific behavior is explicit:
       - use tran_flag for observed transiting planets
       - use observed transit depth if pl_trandep exists
       - handle missing sy_kepmag without falsely killing thousands of NASA rows

This is still a toy / Kepler-ish model. It does not yet use official DR25
one-sigma-depth FITS files, window functions, or injection-recovery efficiency.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    from lifesim.core.data import Data
except Exception:  # lets this file import outside the lifesim environment
    Data = None


class KeplerData:
    # ============================================================
    # Constants and unit conversions
    # ============================================================
    R_SUN_IN_AU = 0.00465047
    R_EARTH_IN_AU = 4.26352e-5
    R_SUN_IN_R_EARTH = 109.076

    # Kepler TPS searched these pulse/transit durations.
    # Sources: NASA Kepler Stellar Table docs; KeplerPORTs.py.
    KEPLER_TPS_DURATIONS_HR = np.array([
        1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0,
        6.0, 7.5, 9.0, 10.5, 12.0, 12.5, 15.0,
    ])

    # KeplerPORTs uses this small factor so approximate MES better matches TPS MES.
    MES_CORRECTION = 1.003

    # Empirical detection-efficiency calibration against the OFFICIAL Kepler DR25
    # pipeline MES (koi_max_mult_ev).  Even with real per-target rrmscdpp* noise,
    # the idealized boxcar formula (geometric depth x sqrt(N) / CDPP) runs hot:
    # script 47 measured median toy/official MES ~ 1.19 (optimistic), so the toy
    # detector crossed MES>=7.1 too readily and over-detected marginal planets.
    # This factor (~1/1.19) folds in the matched-filter-vs-boxcar shape loss and
    # limb-darkening reduction of the *effective* depth, bringing the median
    # toy/official ratio to ~1.0.  Re-derive with scripts/Detector_Calibration/03_kepler_calibration.py.
    MES_OFFICIAL_CALIBRATION = 0.84

    # Earth-size planet across Sun-size star gives roughly 84 ppm.
    # This is geometry, not a universal Kepler detection threshold.
    EARTH_SUN_TRANSIT_DEPTH_PPM = 84.0

    def __init__(
        self,
        data: Union[pd.DataFrame, object],
        source: str = "auto",
        validate_for_detection: bool = True,
        mission_duration_days: float = 4 * 365.25,

        # Kepler-ish detector settings.
        min_transits: int = 3,
        mes_threshold: float = 7.1,
        # Multiplicative calibration of toy MES to the official DR25 pipeline MES.
        # Default = MES_OFFICIAL_CALIBRATION (~1/1.19). Pass 1.0 to recover the
        # raw, uncalibrated (optimistic) boxcar MES, e.g. for the before/after
        # comparison in scripts/Detector_Calibration/03_kepler_calibration.py.
        mes_calibration: Optional[float] = None,
        kepler_mag_limit: float = 16.0,
        fallback_cdpp_ppm: float = 100.0,
        use_kepmag_cdpp_fallback: bool = True,
        cdpp_kp_ref_mag: float = 12.0,
        cdpp_min_ppm: float = 20.0,
        cdpp_max_ppm: float = 2000.0,
        # Intrinsic stellar variability + residual-systematics noise floor,
        # added in QUADRATURE to the magnitude-scaled photon term. Photon noise
        # keeps shrinking for bright stars, but a real light curve never drops
        # below the star's own variability. Without this, nearby bright F/G
        # dwarfs (Kp ~7-9 at 60 pc) get driven toward ~0 ppm noise and every
        # transit is "detected", saturating the FGK detected-fraction panels.
        # This is the main knob controlling bright-star detectability.
        # 28 ppm = "between quiet and typical" FGK dwarf: above Kepler's best-case
        # ~20 ppm quiet-G2V spec, below the ~40 ppm of a mildly active dwarf.
        cdpp_variability_ppm: float = 28.0,

        # NASA-specific switches. These are the keywords your NASA runner uses.
        use_observed_transit_flag_for_nasa: bool = True,
        use_observed_transit_depth_for_nasa: bool = True,
        assume_bright_if_kepmag_missing_for_nasa: bool = True,
        estimate_missing_semimajor_axis: bool = True,

        # Detection efficiency model (same logic as TESSData).
        # 'threshold' = hard MES >= mes_threshold step (default; backward-compatible).
        # 'sigmoid'   = logistic probability weight for expected-detection analysis.
        #   Usage: expected_count = kepler_p_detect.sum()  (NOT a Bernoulli draw)
        detection_model: str = "threshold",
        sigmoid_steepness: float = 1.5,

        # Backward-compatibility knobs from older toy versions.
        # They are kept so older scripts do not crash if they still pass them.
        snr_threshold_best: Optional[float] = None,
        snr_threshold_worst: Optional[float] = None,
        default_noise_ppm: Optional[float] = None,
        depth_threshold_ppm: Optional[float] = None,
    ):
        if Data is not None and isinstance(data, Data):
            self.catalog = pd.DataFrame(data.catalog)
        elif isinstance(data, pd.DataFrame):
            self.catalog = data.copy()
        else:
            self.catalog = pd.DataFrame(data)

        if self.catalog is None or not isinstance(self.catalog, pd.DataFrame):
            raise ValueError("Catalog must be a valid pandas DataFrame.")

        self.source = self._infer_source(source)
        self.mission_duration_days = mission_duration_days
        self.min_transits = min_transits
        self.mes_threshold = mes_threshold
        self.mes_calibration = (
            float(self.MES_OFFICIAL_CALIBRATION) if mes_calibration is None
            else float(mes_calibration)
        )
        self.kepler_mag_limit = kepler_mag_limit

        # If an old caller passes default_noise_ppm, use it as the CDPP fallback.
        if default_noise_ppm is not None and fallback_cdpp_ppm == 100.0:
            fallback_cdpp_ppm = float(default_noise_ppm)
        self.fallback_cdpp_ppm = fallback_cdpp_ppm
        self.use_kepmag_cdpp_fallback = use_kepmag_cdpp_fallback
        self.cdpp_kp_ref_mag = cdpp_kp_ref_mag
        self.cdpp_min_ppm = cdpp_min_ppm
        self.cdpp_max_ppm = cdpp_max_ppm
        self.cdpp_variability_ppm = float(cdpp_variability_ppm)

        self.detection_model = detection_model.lower().strip()
        self.sigmoid_steepness = float(sigmoid_steepness)

        self.use_observed_transit_flag_for_nasa = use_observed_transit_flag_for_nasa
        self.use_observed_transit_depth_for_nasa = use_observed_transit_depth_for_nasa
        self.assume_bright_if_kepmag_missing_for_nasa = assume_bright_if_kepmag_missing_for_nasa
        self.estimate_missing_semimajor_axis = estimate_missing_semimajor_axis

        # Store old parameters only as diagnostics/backward compatibility.
        self.snr_threshold_best = snr_threshold_best
        self.snr_threshold_worst = snr_threshold_worst
        self.default_noise_ppm = default_noise_ppm
        self.depth_threshold_ppm = depth_threshold_ppm

        # Make outside datasets speak the local project language.
        self.catalog = self.standardize_catalog_columns(self.catalog, self.source)
        self._add_basic_helper_columns()

        if self.estimate_missing_semimajor_axis:
            self._estimate_missing_semimajor_axis_if_possible()

        if validate_for_detection:
            self._validate_detection_columns()

    # ============================================================
    # Source inference and column standardization
    # ============================================================

    def _infer_source(self, source: str) -> str:
        source = str(source).lower().strip()
        if source != "auto":
            return source

        cols = set(self.catalog.columns)
        if {"pl_name", "pl_rade", "pl_insol"}.issubset(cols):
            return "pscomppars"
        if {"kepoi_name", "koi_prad", "koi_insol"}.issubset(cols):
            return "koi"
        return "ppop"

    @staticmethod
    def standardize_catalog_columns(df: pd.DataFrame, source: str = "auto") -> pd.DataFrame:
        """
        Convert NASA / KOI / P-Pop-ish columns into shared internal names.

        Internal names used by the detector and plotting scripts:
            mass_p       Earth masses
            radius_p     Earth radii
            flux_p       Earth insolation units
            radius_s     Solar radii
            semimajor_p  AU
            p_orb        days
            inc_p        degrees if available
            kepmag       Kepler magnitude if available
        """
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

        # Helpful P-Pop / older-project aliases, only used when internal column missing.
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

        elif source == "auto":
            df["dataset_source"] = "unknown_auto"

        else:
            raise ValueError("source must be one of: auto, ppop, pscomppars, nasa, ps, koi")

        # Apply generic aliases after source-specific renaming.
        rename_map = {k: v for k, v in generic_map.items() if k in df.columns and v not in df.columns}
        if rename_map:
            df = df.rename(columns=rename_map)

        # PSCompPars st_lum is log10(L/Lsun). Convert to L/Lsun if present.
        if "st_lum_log10" in df.columns and "l_sun" not in df.columns:
            st_lum = pd.to_numeric(df["st_lum_log10"], errors="coerce")
            df["l_sun"] = 10 ** st_lum
            df["l_sun_source"] = "10**st_lum_from_PSCompPars"

        # Make important columns numeric.
        numeric_cols = [
            "mass_p", "radius_p", "flux_p", "radius_s", "mass_s", "teff_s",
            "semimajor_p", "p_orb", "inc_p", "kepmag", "gaiamag", "distance_s",
            "observed_transit_depth_ppm", "observed_transit_depth_percent",
            "observed_transit_duration_hr", "l_sun",
            "rrmscdpp01p5", "rrmscdpp02p0", "rrmscdpp02p5", "rrmscdpp03p0",
            "rrmscdpp03p5", "rrmscdpp04p5", "rrmscdpp05p0", "rrmscdpp06p0",
            "rrmscdpp07p5", "rrmscdpp09p0", "rrmscdpp10p5", "rrmscdpp12p0",
            "rrmscdpp12p5", "rrmscdpp15p0", "dataspan", "dutycycle",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # NASA PSCompPars transit depth is percent; convert percent to ppm.
        if "observed_transit_depth_percent" in df.columns:
            depth_percent = pd.to_numeric(df["observed_transit_depth_percent"], errors="coerce")
            # If observed_transit_depth_ppm already exists, keep it where non-null.
            depth_ppm_from_percent = depth_percent * 1e4
            if "observed_transit_depth_ppm" in df.columns:
                existing = pd.to_numeric(df["observed_transit_depth_ppm"], errors="coerce")
                df["observed_transit_depth_ppm"] = existing.fillna(depth_ppm_from_percent)
            else:
                df["observed_transit_depth_ppm"] = depth_ppm_from_percent

        return df

    def _add_basic_helper_columns(self) -> None:
        """Add useful columns that both P-Pop and NASA plotting scripts expect."""
        # Spectral type from stellar temperature if stype is missing.
        if "stype" not in self.catalog.columns:
            if "teff_s" in self.catalog.columns:
                self.catalog["stype"] = self.catalog["teff_s"].apply(self._stellar_type_from_teff)
            else:
                self.catalog["stype"] = "Unknown"

        # Habitable is not central here, but some old plotting paths expect it.
        if "habitable" not in self.catalog.columns:
            if "flux_p" in self.catalog.columns:
                flux = pd.to_numeric(self.catalog["flux_p"], errors="coerce")
                self.catalog["habitable"] = (flux >= 0.25) & (flux <= 2.0)
            else:
                self.catalog["habitable"] = False

        # Some old plotting code expects luminosity_s and temp_s.
        if "l_sun" in self.catalog.columns and "luminosity_s" not in self.catalog.columns:
            self.catalog["luminosity_s"] = self.catalog["l_sun"]
        if "teff_s" in self.catalog.columns and "temp_s" not in self.catalog.columns:
            self.catalog["temp_s"] = self.catalog["teff_s"]

    @staticmethod
    def _stellar_type_from_teff(teff) -> str:
        if pd.isna(teff):
            return "Unknown"
        teff = float(teff)
        if teff >= 7500:
            return "A"
        if teff >= 6000:
            return "F"
        if teff >= 5200:
            return "G"
        if teff >= 3700:
            return "K"
        if teff > 0:
            return "M"
        return "Unknown"

    def _estimate_missing_semimajor_axis_if_possible(self) -> None:
        """
        Estimate semi-major axis from period and stellar mass when semimajor_p is missing.

        Kepler's third law in Solar units:
            a_AU ≈ (M_star * P_year^2)^(1/3)
        """
        if "semimajor_p" not in self.catalog.columns:
            self.catalog["semimajor_p"] = np.nan

        missing_a = self.catalog["semimajor_p"].isna()
        if not missing_a.any():
            return

        if "p_orb" not in self.catalog.columns:
            return

        p_days = pd.to_numeric(self.catalog["p_orb"], errors="coerce")
        if "mass_s" in self.catalog.columns:
            mstar = pd.to_numeric(self.catalog["mass_s"], errors="coerce").fillna(1.0)
            mass_source = "mass_s"
        else:
            mstar = pd.Series(1.0, index=self.catalog.index)
            mass_source = "assumed_1Msun"

        p_year = p_days / 365.25
        a_est = (mstar * p_year ** 2) ** (1.0 / 3.0)

        self.catalog.loc[missing_a, "semimajor_p"] = a_est.loc[missing_a]
        self.catalog["semimajor_p_source"] = np.where(
            missing_a,
            f"estimated_from_period_and_{mass_source}",
            "catalog",
        )

    def _validate_detection_columns(self) -> None:
        required = ["radius_p", "radius_s", "semimajor_p", "p_orb"]
        missing = [col for col in required if col not in self.catalog.columns]
        if missing:
            raise ValueError(
                "Missing required columns for Kepler detection: "
                + str(missing)
                + "\nIf this is NASA PSCompPars and you only want plotting, use "
                + "KeplerData(df, source='pscomppars', validate_for_detection=False)."
            )

        bad = []
        for col in required:
            values = pd.to_numeric(self.catalog[col], errors="coerce")
            if values.notna().sum() == 0:
                bad.append(col)
        if bad:
            raise ValueError(f"Detection columns exist but are entirely NaN: {bad}")

    def _is_nasa_like(self) -> bool:
        return self.source in {"pscomppars", "ps", "nasa", "koi"} or str(
            self.catalog.get("dataset_source", pd.Series([""])).iloc[0]
        ).startswith("NASA")

    def _find_first_existing_column(self, possible_names):
        for name in possible_names:
            if name in self.catalog.columns:
                return name
        return None

    # ============================================================
    # Transit / depth / brightness calculations
    # ============================================================

    def calc_transit_depth_fraction(self):
        r_planet_rearth = pd.to_numeric(self.catalog["radius_p"], errors="coerce")
        r_star_rearth = pd.to_numeric(self.catalog["radius_s"], errors="coerce") * self.R_SUN_IN_R_EARTH
        return (r_planet_rearth / r_star_rearth) ** 2

    def calc_transit_depth_ppm_model(self):
        return self.calc_transit_depth_fraction() * 1e6

    # Backward-compatible name.
    def calc_transit_depth_ppm(self):
        if self._is_nasa_like() and self.use_observed_transit_depth_for_nasa:
            if "observed_transit_depth_ppm" in self.catalog.columns:
                observed = pd.to_numeric(self.catalog["observed_transit_depth_ppm"], errors="coerce")
                model = self.calc_transit_depth_ppm_model()
                depth = observed.fillna(model)
                self.catalog["transit_depth_source"] = np.where(observed.notna(), "observed", "model_Rp_Rstar")
                return depth
        self.catalog["transit_depth_source"] = "model_Rp_Rstar"
        return self.calc_transit_depth_ppm_model()

    def calc_transiting_from_inclination(self):
        """
        For P-Pop:
            use geometric inclination.

        For NASA PSCompPars/KOI:
            if requested and tran_flag exists, use tran_flag because the dataset is already observed transiting planets.
        """
        if self._is_nasa_like() and self.use_observed_transit_flag_for_nasa and "tran_flag" in self.catalog.columns:
            tran_flag = pd.to_numeric(self.catalog["tran_flag"], errors="coerce").fillna(0)
            transiting = tran_flag.astype(int) == 1
            self.catalog["impact_parameter_toy"] = np.nan
            self.catalog["transiting_source"] = "tran_flag"
            return transiting.fillna(False)

        inc_col = self._find_first_existing_column(["inc_p", "inclination", "pl_orbincl", "koi_incl"])
        if inc_col is None:
            raise ValueError(
                "No inclination column found. Need inc_p for P-Pop geometry, "
                "or tran_flag for NASA observed transiting planets."
            )

        inc = pd.to_numeric(self.catalog[inc_col], errors="coerce")
        inc_rad = inc if inc.max(skipna=True) <= 3.2 else np.deg2rad(inc)

        a_au = pd.to_numeric(self.catalog["semimajor_p"], errors="coerce")
        r_star_au = pd.to_numeric(self.catalog["radius_s"], errors="coerce") * self.R_SUN_IN_AU
        r_planet_au = pd.to_numeric(self.catalog["radius_p"], errors="coerce") * self.R_EARTH_IN_AU

        b = (a_au * np.abs(np.cos(inc_rad))) / r_star_au
        transiting = b <= (1 + r_planet_au / r_star_au)

        self.catalog["impact_parameter_toy"] = b
        self.catalog["transiting_source"] = "inc_p_geometry"
        return transiting.fillna(False)

    def calc_star_brightness_proxy(self):
        """Fallback only. Prefer real Kepler magnitude."""
        required = ["l_sun", "distance_s"]
        missing = [col for col in required if col not in self.catalog.columns]
        if missing:
            raise ValueError(
                f"Missing columns for brightness proxy: {missing}. Prefer using real kepmag."
            )

        l_sun = pd.to_numeric(self.catalog["l_sun"], errors="coerce").clip(lower=1e-12)
        distance_s = pd.to_numeric(self.catalog["distance_s"], errors="coerce").clip(lower=1e-12)

        star_flux_proxy = l_sun / (distance_s ** 2)
        approx_mbol = 4.74 - 2.5 * np.log10(l_sun) + 5 * np.log10(distance_s / 10)
        return star_flux_proxy, approx_mbol

    def calc_bright_enough_kepler(self):
        """
        Prefer 8 <= kepmag <= 16.

        For NASA rows with missing sy_kepmag, optionally assume bright enough, because otherwise
        missing metadata would masquerade as a failed Kepler detection.
        """
        if "kepmag" in self.catalog.columns:
            mag = pd.to_numeric(self.catalog["kepmag"], errors="coerce")
            bright = mag <= self.kepler_mag_limit

            if self._is_nasa_like() and self.assume_bright_if_kepmag_missing_for_nasa:
                bright = bright | mag.isna()
                source = np.where(mag.notna(), "kepmag", "missing_kepmag_assumed_bright_for_nasa")
            else:
                source = np.where(mag.notna(), "kepmag", "missing_kepmag_not_bright")

            self.catalog["kepler_mag_used"] = mag
            self.catalog["kepler_mag_source"] = source
            self.catalog["bright_enough_kepler"] = bright.fillna(False)
            return self.catalog["bright_enough_kepler"]

        # P-Pop fallback: approximate bolometric magnitude if possible.
        if "l_sun" in self.catalog.columns and "distance_s" in self.catalog.columns:
            star_flux_proxy, approx_mbol = self.calc_star_brightness_proxy()
            bright = approx_mbol <= self.kepler_mag_limit
            self.catalog["star_flux_proxy"] = star_flux_proxy
            self.catalog["approx_mbol"] = approx_mbol
            self.catalog["kepler_mag_used"] = approx_mbol
            self.catalog["kepler_mag_source"] = "approx_mbol_proxy"
            self.catalog["bright_enough_kepler"] = bright.fillna(False)
            return self.catalog["bright_enough_kepler"]

        # Last resort: let the detector run, but label it clearly.
        bright = pd.Series(True, index=self.catalog.index)
        self.catalog["kepler_mag_used"] = np.nan
        self.catalog["kepler_mag_source"] = "missing_mag_assumed_bright_last_resort"
        self.catalog["bright_enough_kepler"] = bright
        return bright

    # ============================================================
    # Kepler-ish MES/depth significance
    # ============================================================

    def estimate_transit_duration_hours(self):
        # Prefer observed duration if NASA provides it.
        if self._is_nasa_like() and "observed_transit_duration_hr" in self.catalog.columns:
            observed = pd.to_numeric(self.catalog["observed_transit_duration_hr"], errors="coerce")
        else:
            observed = pd.Series(np.nan, index=self.catalog.index)

        p_days = pd.to_numeric(self.catalog["p_orb"], errors="coerce")
        r_star_au = pd.to_numeric(self.catalog["radius_s"], errors="coerce") * self.R_SUN_IN_AU
        a_au = pd.to_numeric(self.catalog["semimajor_p"], errors="coerce")

        model = (p_days * 24.0 / np.pi) * (r_star_au / a_au)
        duration_hr = observed.fillna(model)
        duration_hr = duration_hr.replace([np.inf, -np.inf], np.nan).fillna(6.0)
        duration_hr = duration_hr.clip(
            lower=self.KEPLER_TPS_DURATIONS_HR.min(),
            upper=self.KEPLER_TPS_DURATIONS_HR.max(),
        )

        self.catalog["transit_duration_hr_est"] = duration_hr
        self.catalog["transit_duration_source"] = np.where(observed.notna(), "observed", "model")
        return duration_hr

    def nearest_kepler_duration_label(self, duration_hr):
        durations = self.KEPLER_TPS_DURATIONS_HR
        arr = np.asarray(duration_hr, dtype=float)
        idx = np.abs(arr[:, None] - durations[None, :]).argmin(axis=1)
        nearest = durations[idx]

        labels = []
        for d in nearest:
            whole = int(np.floor(d))
            frac = int(round((d - whole) * 10))
            labels.append(f"{whole:02d}p{frac}")

        self.catalog["nearest_kepler_duration_hr"] = nearest
        self.catalog["nearest_kepler_duration_label"] = labels
        return labels

    def estimate_cdpp_from_kepler_mag(self):
        """
        Minimal photon-noise-like CDPP fallback.

        Real Kepler rows should use rrmscdppXXpX columns when available.
        For P-Pop/Gaia rows with no real CDPP columns, this makes faint
        stars noisier instead of using one flat fallback CDPP for everything.

        The scaling is intentionally simple:
            CDPP(Kp) = fallback_cdpp_ppm * 10**(0.2 * (Kp - cdpp_kp_ref_mag))
        because photon noise grows approximately with sqrt(1/flux).
        """
        if not self.use_kepmag_cdpp_fallback:
            return pd.Series(float(self.fallback_cdpp_ppm), index=self.catalog.index)

        if "kepler_mag_used" in self.catalog.columns:
            mag = pd.to_numeric(self.catalog["kepler_mag_used"], errors="coerce")
        elif "kepmag" in self.catalog.columns:
            mag = pd.to_numeric(self.catalog["kepmag"], errors="coerce")
        elif "l_sun" in self.catalog.columns and "distance_s" in self.catalog.columns:
            _, mag = self.calc_star_brightness_proxy()
        else:
            mag = pd.Series(np.nan, index=self.catalog.index)

        cdpp_photon = self.fallback_cdpp_ppm * 10 ** (0.2 * (mag - self.cdpp_kp_ref_mag))
        cdpp_photon = cdpp_photon.replace([np.inf, -np.inf], np.nan).fillna(self.fallback_cdpp_ppm)

        # Add the stellar-variability / residual-systematics floor in QUADRATURE.
        # Photon noise -> 0 for very bright stars, but a real light curve never
        # drops below the star's own variability. This stops nearby bright F/G
        # dwarfs (Kp ~7-9 at 60 pc) from becoming effectively noiseless and
        # making every transit detectable. M dwarfs (Kp ~13, CDPP ~200 ppm) are
        # essentially unchanged, so the M-dwarf science is unaffected.
        cdpp = np.sqrt(cdpp_photon ** 2 + self.cdpp_variability_ppm ** 2)
        return cdpp.clip(lower=self.cdpp_min_ppm, upper=self.cdpp_max_ppm)

    def get_cdpp_ppm_for_duration(self, duration_hr):
        labels = self.nearest_kepler_duration_label(duration_hr)
        fallback_from_mag = self.estimate_cdpp_from_kepler_mag()
        cdpp_values = []
        cdpp_sources = []

        for i, label in enumerate(labels):
            col = f"rrmscdpp{label}"
            if col in self.catalog.columns:
                val = pd.to_numeric(pd.Series([self.catalog.iloc[i][col]]), errors="coerce").iloc[0]
                if pd.notna(val):
                    cdpp_values.append(float(val))
                    cdpp_sources.append(col)
                    continue

            cdpp_values.append(float(fallback_from_mag.iloc[i]))
            if self.use_kepmag_cdpp_fallback:
                cdpp_sources.append("kepmag_cdpp_fallback")
            else:
                cdpp_sources.append("fallback_cdpp_ppm")

        cdpp_ppm = pd.Series(cdpp_values, index=self.catalog.index, dtype=float)
        self.catalog["kepler_cdpp_ppm"] = cdpp_ppm
        self.catalog["kepler_cdpp_source"] = cdpp_sources
        return cdpp_ppm

    def calc_number_of_observed_transits_keplerish(self):
        p_orb_days = pd.to_numeric(self.catalog["p_orb"], errors="coerce").replace(0, np.nan)

        if "dataspan" in self.catalog.columns:
            dataspan = pd.to_numeric(self.catalog["dataspan"], errors="coerce").fillna(self.mission_duration_days)
        else:
            dataspan = pd.Series(self.mission_duration_days, index=self.catalog.index)

        if "dutycycle" in self.catalog.columns:
            dutycycle = pd.to_numeric(self.catalog["dutycycle"], errors="coerce").fillna(1.0)
        else:
            dutycycle = pd.Series(1.0, index=self.catalog.index)

        n_transits = np.floor(dataspan * dutycycle / p_orb_days)
        n_transits = n_transits.replace([np.inf, -np.inf], np.nan).fillna(0)
        n_transits = np.maximum(n_transits, 0)

        self.catalog["kepler_dataspan_used_days"] = dataspan
        self.catalog["kepler_dutycycle_used"] = dutycycle
        self.catalog["n_transits_keplerish"] = n_transits
        return n_transits

    # Backward-compatible old name.
    def calc_number_of_transits(self):
        return self.calc_number_of_observed_transits_keplerish()

    def calc_kepler_mes(self):
        transit_depth_ppm = self.calc_transit_depth_ppm()
        duration_hr = self.estimate_transit_duration_hours()
        cdpp_ppm = self.get_cdpp_ppm_for_duration(duration_hr)
        n_transits = self.calc_number_of_observed_transits_keplerish()

        sqrt_n = np.sqrt(n_transits.clip(lower=1))
        mes = (transit_depth_ppm / cdpp_ppm.replace(0, np.nan) * sqrt_n
               * self.MES_CORRECTION * self.mes_calibration)
        mes = mes.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        one_sigma_depth_ppm = cdpp_ppm / (sqrt_n * self.MES_CORRECTION * self.mes_calibration)
        min_detectable_depth_ppm = self.mes_threshold * one_sigma_depth_ppm

        self.catalog["transit_depth_ppm"] = transit_depth_ppm
        self.catalog["kepler_mes"] = mes
        self.catalog["kepler_mes_threshold"] = self.mes_threshold
        self.catalog["one_sigma_depth_ppm"] = one_sigma_depth_ppm
        self.catalog["min_detectable_depth_ppm"] = min_detectable_depth_ppm
        return mes

    # Backward-compatible old method name.
    def calc_snr(self):
        return self.calc_kepler_mes()

    def calc_depth_good_keplerish(self):
        mes = self.calc_kepler_mes()
        n_transits = self.catalog["n_transits_keplerish"]
        depth_good = (n_transits >= self.min_transits) & (mes >= self.mes_threshold)

        self.catalog["kepler_enough_transits"] = n_transits >= self.min_transits
        self.catalog["kepler_depth_good"] = depth_good.fillna(False)
        return self.catalog["kepler_depth_good"]

    def add_miss_reason_category(self):
        """Explain why each row passes or fails the current toy Kepler detector."""
        transiting = self.catalog["transiting_geometric"].astype(bool)
        bright = self.catalog["kepler_star_bright_enough"].astype(bool)
        enough = self.catalog["kepler_enough_transits"].astype(bool)
        mes = pd.to_numeric(self.catalog["kepler_mes"], errors="coerce").fillna(0)
        detected = self.catalog["detected"].astype(bool)

        reason = np.full(len(self.catalog), "other_missed", dtype=object)
        reason[~transiting.to_numpy()] = "not_transiting"
        reason[(transiting & ~bright).to_numpy()] = "host_star_too_faint"
        reason[(transiting & bright & ~enough).to_numpy()] = "too_few_transits"
        reason[(transiting & bright & enough & (mes < self.mes_threshold)).to_numpy()] = "too_shallow_or_low_mes"
        reason[detected.to_numpy()] = "detected"

        self.catalog["reason_category"] = reason
        self.catalog["miss_reason"] = reason
        return self.catalog["reason_category"]

    # ============================================================
    # Final detector
    # ============================================================

    def determine_detectable(self):
        """
        Kepler-ish detector:
            detected = transiting and bright_enough_kepler and depth_good
        """
        self._validate_detection_columns()

        transiting = self.calc_transiting_from_inclination()
        transit_depth_fraction = self.calc_transit_depth_fraction()
        transit_depth_ppm = self.calc_transit_depth_ppm()
        bright_enough_kepler = self.calc_bright_enough_kepler()
        depth_good = self.calc_depth_good_keplerish()

        detected = transiting & bright_enough_kepler & depth_good

        # Detection probability weight — NOT a Bernoulli draw.
        # kepler_p_detect is 0/1 for threshold; smooth logistic for sigmoid.
        # Gated by transiting, bright, and enough-transits so the MES-based
        # probability only applies to planets that pass all upstream gates.
        # Usage: expected_detected_in_bin = kepler_p_detect[mask].sum()
        mes_col = pd.to_numeric(self.catalog["kepler_mes"], errors="coerce").fillna(0.0)
        enough = self.catalog["kepler_enough_transits"].astype(bool)
        all_gates = transiting.astype(bool) & bright_enough_kepler.astype(bool) & enough
        if self.detection_model == "sigmoid":
            p_mes = 1.0 / (1.0 + np.exp(-self.sigmoid_steepness * (mes_col - self.mes_threshold)))
        else:
            p_mes = (mes_col >= self.mes_threshold).astype(float)
        self.catalog["kepler_p_detect"] = p_mes.where(all_gates, 0.0)

        self.catalog["transiting_geometric"] = transiting
        self.catalog["transit_depth_fraction"] = transit_depth_fraction
        self.catalog["transit_depth_ppm"] = transit_depth_ppm
        self.catalog["kepler_star_bright_enough"] = bright_enough_kepler
        self.catalog["kepler_depth_pass"] = depth_good
        self.catalog["detected"] = detected

        # Backward-compatible names for older plotting files.
        self.catalog["kepler_star_bright_enough_best"] = bright_enough_kepler
        self.catalog["kepler_star_bright_enough_worst"] = bright_enough_kepler
        self.catalog["kepler_depth_pass_best"] = depth_good
        self.catalog["kepler_depth_pass_worst"] = depth_good
        self.catalog["detected_best"] = detected
        self.catalog["detected_worst"] = detected

        self.add_miss_reason_category()

        return self.catalog


# ============================================================
# Future upgrades: keep commented for later
# ============================================================

# FUTURE UPGRADE 1:
# Replace CDPP * sqrt(N_transits) with official DR25 one-sigma depth FITS products.
# KeplerPORTs idea:
#     MES = transit_depth_ppm / one_sigma_depth_ppm * 1.003
# This is better because one-sigma depth includes period-dependent missing data and target-specific noise.

# UPGRADE 2 (implemented):
# Probabilistic detection efficiency via detection_model='sigmoid'.
# kepler_p_detect = logistic(sigmoid_steepness * (MES - mes_threshold))
# Used as a weight (sum(p_i) per bin), not a Bernoulli draw.
# For a fuller treatment: replace with KeplerPORTs DEMod.final_detEffs(MES, period),
# which is period-dependent and accounts for window function probability.

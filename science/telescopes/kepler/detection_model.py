"""Kepler toy transit detector for P-Pop catalogs and NASA PSCompPars/KOI tables, e.g.
KeplerData(df, source='pscomppars').determine_detectable(). A planet is detected if it transits,
is bright enough and passes the MES threshold. No DR25 one-sigma-depth maps or window functions.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from science.physics import infer_stellar_type
from science.telescopes import transit_shape

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

    # Measured duration dependence of Kepler noise: median rrmscdpp(T) / rrmscdpp(6 h) over the DR25
    # KOI hosts in data/exoplanet_csv/koi_cumulative_stellar.csv. Redder than white noise (T^-0.5).
    # Scales the magnitude-based fallback CDPP, which is a 6.5-hr value, to the transit's T14.
    CDPP_SCALING_HR = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0, 6.0, 7.5, 9.0, 10.5, 12.0, 12.5, 15.0])
    CDPP_SCALING = np.array([1.607, 1.435, 1.358, 1.264, 1.206, 1.104, 1.062, 1.000, 0.933,
                             0.882, 0.845, 0.817, 0.810, 0.772])
    FALLBACK_CDPP_DURATION_HR = 6.5

    # Earth-size planet across Sun-size star gives roughly 84 ppm.
    # This is geometry, not a universal Kepler detection threshold.
    EARTH_SUN_TRANSIT_DEPTH_PPM = 84.0

    # Fallback noise for rows without per-target rrmscdpp (every P-Pop row): 6.5-hr CDPP of
    # 12th-mag Kepler dwarfs. Gilliland et al. 2011 (ApJS 197, 6) measured a median intrinsic
    # stellar noise of 19.5 ppm; Gilliland et al. 2015 (AJ 150, 133) found it unchanged over the
    # full mission, with stellar + Poisson = 664 ppm^2 and a 98 ppm^2 instrument/pipeline residual
    # in the final SOC 9.2 processing. The non-stellar part is sqrt(664 - 19.5^2 + 98) = 19.5 ppm,
    # so CDPP(Kp=12) = 19.5 (+) 19.5 = 28 ppm, matching the ~30 ppm those papers quote.
    CDPP_NONSTELLAR_KP12_PPM = 19.5
    CDPP_STELLAR_PPM = 19.5

    def __init__(
        self,
        data: Union[pd.DataFrame, object],
        source: str = "auto",
        validate_for_detection: bool = True,
        mission_duration_days: float = 4 * 365.25,

        # Kepler-ish detector settings.
        min_transits: int = 3,
        mes_threshold: float = 7.1,
        kepler_mag_limit: float = 16.0,
        fallback_cdpp_ppm: float = CDPP_NONSTELLAR_KP12_PPM,
        use_kepmag_cdpp_fallback: bool = True,
        cdpp_kp_ref_mag: float = 12.0,
        cdpp_min_ppm: float = 20.0,
        cdpp_max_ppm: float = 2000.0,
        # Stellar-variability floor added in quadrature to photon noise, so bright nearby F/G dwarfs
        # aren't treated as noiseless (which saturates FGK detections). Main knob for bright-star
        # detectability; default is the Gilliland et al. median (see CDPP_STELLAR_PPM).
        cdpp_variability_ppm: float = CDPP_STELLAR_PPM,

        # NASA-specific switches. These are the keywords your NASA runner uses.
        use_observed_transit_flag_for_nasa: bool = True,
        use_observed_transit_depth_for_nasa: bool = True,
        assume_bright_if_kepmag_missing_for_nasa: bool = True,
        estimate_missing_semimajor_axis: bool = True,

        # Detection efficiency model (same logic as TESSData).
        # 'threshold' = hard MES >= mes_threshold step (default).
        # 'sigmoid'   = logistic probability weight for expected-detection analysis.
        #   Usage: expected_count = kepler_p_detect.sum()  (NOT a Bernoulli draw)
        detection_model: str = "threshold",
        sigmoid_steepness: float = 1.5,
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
        self.kepler_mag_limit = kepler_mag_limit
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

        # Make outside datasets speak the local project language.
        self.catalog = self.standardize_catalog_columns(self.catalog, self.source)
        self.nasa_like_source = self.source in {"pscomppars", "ps", "nasa", "koi"}
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
    def standardize_catalog_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Rename NASA / KOI / P-Pop columns to the shared internal names: mass_p, radius_p [Earth],
        flux_p [Earth insolation], radius_s [Sun], semimajor_p [AU], p_orb [d], inc_p [deg], kepmag.
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

        else:
            raise ValueError("source must be one of: ppop, pscomppars, nasa, ps, koi")

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
                self.catalog["stype"] = infer_stellar_type(self.catalog["teff_s"])
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

    def _estimate_missing_semimajor_axis_if_possible(self) -> None:
        """Fill missing semimajor_p from period and stellar mass: a_AU = (M_star * P_yr^2)^(1/3)."""
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

    # ============================================================
    # Transit / depth / brightness calculations
    # ============================================================

    def calc_transit_depth_fraction(self):
        r_planet_rearth = pd.to_numeric(self.catalog["radius_p"], errors="coerce")
        r_star_rearth = pd.to_numeric(self.catalog["radius_s"], errors="coerce") * self.R_SUN_IN_R_EARTH
        return (r_planet_rearth / r_star_rearth) ** 2

    # Observed depth for NASA rows that have one, else the (Rp/R*)^2 model depth.
    def calc_transit_depth_ppm(self):
        if self.nasa_like_source and self.use_observed_transit_depth_for_nasa:
            if "observed_transit_depth_ppm" in self.catalog.columns:
                observed = pd.to_numeric(self.catalog["observed_transit_depth_ppm"], errors="coerce")
                model = self.calc_transit_depth_fraction() * 1e6
                depth = observed.fillna(model)
                self.catalog["transit_depth_source"] = np.where(observed.notna(), "observed", "model_Rp_Rstar")
                return depth
        self.catalog["transit_depth_source"] = "model_Rp_Rstar"
        return self.calc_transit_depth_fraction() * 1e6

    def calc_transiting_from_inclination(self):
        """P-Pop: use geometric inclination. NASA PSCompPars/KOI: optionally use tran_flag,
        since those planets are already observed to transit.
        """
        b = self._impact_from_inclination()
        self.catalog["impact_parameter"] = b
        if self.nasa_like_source and self.use_observed_transit_flag_for_nasa and "tran_flag" in self.catalog.columns:
            tran_flag = pd.to_numeric(self.catalog["tran_flag"], errors="coerce").fillna(0)
            self.catalog["transiting_source"] = "tran_flag"
            return (tran_flag.astype(int) == 1).fillna(False)
        if b.isna().all():
            raise ValueError(
                "No inclination column found. Need inc_p for P-Pop geometry, "
                "or tran_flag for NASA observed transiting planets."
            )
        self.catalog["transiting_source"] = "inc_p_geometry"
        return (b <= 1 + self._radius_ratio()).fillna(False)

    def _radius_ratio(self) -> pd.Series:
        """k = Rp / R*: fitted ratio (KOI), else the radii, else a measured depth."""
        rp = pd.to_numeric(self.catalog["radius_p"], errors="coerce")
        rs = pd.to_numeric(self.catalog["radius_s"], errors="coerce") * self.R_SUN_IN_R_EARTH
        k = pd.to_numeric(self.catalog.get("observed_radius_ratio", pd.Series(np.nan, index=self.catalog.index)), errors="coerce")
        observed = pd.to_numeric(self.catalog.get("observed_transit_depth_ppm", pd.Series(np.nan, index=self.catalog.index)), errors="coerce")
        return k.fillna(rp / rs).fillna(np.sqrt(observed.clip(lower=0) / 1e6))

    def _a_over_rstar(self) -> pd.Series:
        a = pd.to_numeric(self.catalog["semimajor_p"], errors="coerce")
        return a / (pd.to_numeric(self.catalog["radius_s"], errors="coerce") * self.R_SUN_IN_AU)

    def _impact_from_inclination(self) -> pd.Series:
        inc_col = next((c for c in ["inc_p", "inclination", "pl_orbincl", "koi_incl"] if c in self.catalog.columns), None)
        if inc_col is None:
            return pd.Series(np.nan, index=self.catalog.index)
        inc = pd.to_numeric(self.catalog[inc_col], errors="coerce")
        inc_rad = inc if inc.max(skipna=True) <= 3.2 else np.deg2rad(inc)
        return self._a_over_rstar() * np.abs(np.cos(inc_rad))

    def impact_parameter(self) -> pd.Series:
        """b from a measured T14, else the inclination, else (1 + k) / 2, the mean for isotropic
        orbits that transit. Catalogued inclinations come from fits with their own a/R*, so pairing
        them with the stellar a/R* here can put an observed transit off the star."""
        k = self._radius_ratio()
        b = pd.to_numeric(self.catalog["impact_parameter"], errors="coerce")
        observed = self._observed_duration()
        p = pd.to_numeric(self.catalog["p_orb"], errors="coerce")
        from_duration = pd.Series(transit_shape.impact_from_t14(observed, p, self._a_over_rstar(), k),
                                  index=self.catalog.index).where(observed.notna())
        source = np.where(from_duration.notna(), "observed_duration", np.where(b.notna(), "inclination", "mean_for_transiting"))
        b = from_duration.fillna(b).fillna((1 + k) / 2)
        self.catalog["impact_parameter"] = b
        self.catalog["impact_parameter_source"] = source
        return b

    def signal_rms_ppm(self) -> pd.Series:
        """rms of the limb-darkened transit dip over T14 (see transit_shape); what the MES uses."""
        observed = None
        if self.nasa_like_source and self.use_observed_transit_depth_for_nasa and "observed_transit_depth_ppm" in self.catalog.columns:
            observed = pd.to_numeric(self.catalog["observed_transit_depth_ppm"], errors="coerce")
        rms = transit_shape.signal_rms_ppm(self._radius_ratio(), self.catalog["impact_parameter"], "Kepler", observed)
        return pd.Series(rms, index=self.catalog.index)

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
        """Require 8 <= kepmag <= 16. NASA rows missing sy_kepmag can optionally pass, so missing
        metadata isn't counted as a failed detection.
        """
        if "kepmag" in self.catalog.columns:
            mag = pd.to_numeric(self.catalog["kepmag"], errors="coerce")
            bright = mag <= self.kepler_mag_limit

            if self.nasa_like_source and self.assume_bright_if_kepmag_missing_for_nasa:
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

    def _observed_duration(self) -> pd.Series:
        if self.nasa_like_source and "observed_transit_duration_hr" in self.catalog.columns:
            return pd.to_numeric(self.catalog["observed_transit_duration_hr"], errors="coerce")
        return pd.Series(np.nan, index=self.catalog.index)

    def estimate_transit_duration_hours(self):
        """T14 in hours: the observed value for NASA rows, else a circular orbit with impact parameter b."""
        observed = self._observed_duration()
        p_days = pd.to_numeric(self.catalog["p_orb"], errors="coerce")
        model = pd.Series(transit_shape.t14_hours(p_days, self._a_over_rstar(), self._radius_ratio(),
                                                  self.catalog["impact_parameter"]), index=self.catalog.index)
        duration_hr = observed.fillna(model).replace([np.inf, -np.inf], np.nan).fillna(6.0).clip(0.05, 48.0)
        self.catalog["transit_duration_hr_est"] = duration_hr
        self.catalog["transit_duration_source"] = np.where(observed.notna(), "observed", "model_t14")
        return duration_hr

    def estimate_cdpp_from_kepler_mag(self):
        """Photon-noise CDPP fallback for rows without real rrmscdpp columns (e.g. P-Pop/Gaia):
        CDPP(Kp) = fallback_cdpp_ppm * 10**(0.2 * (Kp - cdpp_kp_ref_mag)), so faint stars are noisier.
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

        # Add the stellar-variability floor in quadrature so bright F/G dwarfs (Kp ~7-9 at 60 pc) don't
        # become noiseless. The 10**(0.2 dKp) scaling is photon-limited, so it underestimates noise
        # for Kp >~ 14 where background and read noise matter; within 60 pc that is only M dwarfs.
        cdpp = np.sqrt(cdpp_photon ** 2 + self.cdpp_variability_ppm ** 2)
        return cdpp.clip(lower=self.cdpp_min_ppm, upper=self.cdpp_max_ppm)

    def get_cdpp_ppm_for_duration(self, duration_hr):
        """CDPP at T14: the star's own rrmscdpp columns interpolated log-log (extrapolated past 1.5
        or 15 h), else the magnitude fallback scaled from 6.5 h by the measured CDPP_SCALING."""
        duration_hr = np.asarray(duration_hr, float)
        grid = self.KEPLER_TPS_DURATIONS_HR
        cols = [f"rrmscdpp{int(d):02d}p{int(round((d - int(d)) * 10))}" for d in grid]
        table = np.column_stack([
            pd.to_numeric(self.catalog[c], errors="coerce").to_numpy(float) if c in self.catalog.columns
            else np.full(len(self.catalog), np.nan) for c in cols])
        lo = np.clip(np.searchsorted(np.log(grid), np.log(duration_hr)) - 1, 0, len(grid) - 2)
        w = (np.log(duration_hr) - np.log(grid[lo])) / (np.log(grid[lo + 1]) - np.log(grid[lo]))
        rows = np.arange(len(self.catalog))
        c_lo, c_hi = table[rows, lo], table[rows, lo + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            own = np.exp(np.log(c_lo) + w * (np.log(c_hi) - np.log(c_lo)))
        has_own = np.isfinite(own) & (c_lo > 0) & (c_hi > 0)

        scale = (transit_shape.loglog_interp(duration_hr, self.CDPP_SCALING_HR, self.CDPP_SCALING)
                 / transit_shape.loglog_interp(self.FALLBACK_CDPP_DURATION_HR, self.CDPP_SCALING_HR, self.CDPP_SCALING))
        fallback = self.estimate_cdpp_from_kepler_mag().to_numpy(float) * scale

        cdpp_ppm = pd.Series(np.where(has_own, own, fallback), index=self.catalog.index)
        self.catalog["kepler_cdpp_ppm"] = cdpp_ppm
        self.catalog["kepler_cdpp_source"] = np.where(
            has_own, "rrmscdpp_interpolated",
            "kepmag_cdpp_fallback" if self.use_kepmag_cdpp_fallback else "fallback_cdpp_ppm")
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

    def calc_kepler_mes(self):
        """MES = rms dip over T14 / CDPP(T14) * sqrt(N transits)."""
        transit_depth_ppm = self.calc_transit_depth_ppm()
        signal_ppm = self.signal_rms_ppm()
        duration_hr = self.estimate_transit_duration_hours()
        cdpp_ppm = self.get_cdpp_ppm_for_duration(duration_hr)
        n_transits = self.calc_number_of_observed_transits_keplerish()

        sqrt_n = np.sqrt(n_transits.clip(lower=1))
        mes = (signal_ppm / cdpp_ppm.replace(0, np.nan) * sqrt_n).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        one_sigma_ppm = cdpp_ppm / sqrt_n

        self.catalog["transit_depth_ppm"] = transit_depth_ppm
        self.catalog["signal_rms_ppm"] = signal_ppm
        self.catalog["kepler_mes"] = mes
        self.catalog["kepler_mes_threshold"] = self.mes_threshold
        self.catalog["one_sigma_depth_ppm"] = one_sigma_ppm
        self.catalog["min_detectable_depth_ppm"] = self.mes_threshold * one_sigma_ppm
        return mes

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
        self.impact_parameter()
        transit_depth_fraction = self.calc_transit_depth_fraction()
        transit_depth_ppm = self.calc_transit_depth_ppm()
        bright_enough_kepler = self.calc_bright_enough_kepler()
        depth_good = self.calc_depth_good_keplerish()

        detected = transiting & bright_enough_kepler & depth_good

        # Detection probability weight, not a Bernoulli draw: 0/1 for threshold, logistic for sigmoid,
        # zeroed unless transiting, bright and enough transits. Use kepler_p_detect[mask].sum().
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

        # Kepler has one scenario; the shared plotters read best/worst (real for HWO and LIFEsim).
        self.catalog["detected_best"] = detected
        self.catalog["detected_worst"] = detected

        self.add_miss_reason_category()

        return self.catalog


# ============================================================
# Future upgrades: keep commented for later
# ============================================================

# FUTURE UPGRADE 1: replace CDPP * sqrt(N_transits) with the official DR25 one-sigma depth maps
# (KeplerPORTs: MES = depth_ppm / one_sigma_depth_ppm * 1.003), which include period-dependent gaps.

# UPGRADE 2 (implemented): detection_model='sigmoid' gives
# kepler_p_detect = logistic(sigmoid_steepness * (MES - mes_threshold)), used as a weight.
# Fuller option: KeplerPORTs DEMod.final_detEffs(MES, period).

"""Kepler toy transit detector for P-Pop catalogs and NASA PSCompPars/KOI tables.

KEPLER MODEL

Included for flat-universe simulations:
- min transits: >= 3 observed transits by default
- MES threshold: >= 7.1, the Kepler TPS matched-filter threshold
- transit shape: limb-darkened Kepler-band signal RMS from Claret & Bloemen 2011

Included for calibration against real Kepler planets:
- CDPP: rrmscdpp01p5-rrmscdpp15p0 from the Kepler archive, interpolated to T14
- official MES comparison: plotting/paper_figures/recovery_3x1.py compares kepler_mes
  to koi_max_mult_ev from the NASA KOI cumulative table

Not included:
- DR25 one-sigma-depth maps, period/window completeness, injection recovery, dilution,
  secondary vetting, or other pipeline disposition logic. Crossing 7.1 MES is necessary
  for a Threshold-Crossing Event, not sufficient for planet validation.

Calibration data source:
- NASA KOI cumulative table joined to Kepler stellar CDPP columns
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from science.physics import (
    apparent_bolometric_mag,
)
from science.telescopes.kepler import data_processing
from science.telescopes import transit_shape


class KeplerData:
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
        fallback_cdpp_ppm: float = 19.5,
        use_kepmag_cdpp_fallback: bool = True,
        cdpp_kp_ref_mag: float = 12.0,
        cdpp_min_ppm: float = 20.0,
        cdpp_max_ppm: float = 2000.0,
        # Stellar-variability floor added in quadrature to photon noise, so bright nearby F/G dwarfs
        # aren't treated as noiseless (which saturates FGK detections). Main knob for bright-star
        # detectability; default is the Gilliland et al. median.
        cdpp_variability_ppm: float = 19.5,

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

        # Kepler TPS searched these pulse/transit durations.
        # Sources: NASA Kepler Stellar Table docs; KeplerPORTs.py.
        self.kepler_tps_durations_hr = np.array([
            1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0,
            6.0, 7.5, 9.0, 10.5, 12.0, 12.5, 15.0,
        ])
        # Median rrmscdpp(T) / rrmscdpp(6 h) over DR25 KOI hosts; redder than white noise.
        self.cdpp_scaling_hr = np.array([
            1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.0,
            6.0, 7.5, 9.0, 10.5, 12.0, 12.5, 15.0,
        ])
        self.cdpp_scaling = np.array([
            1.607, 1.435, 1.358, 1.264, 1.206, 1.104, 1.062,
            1.000, 0.933, 0.882, 0.845, 0.817, 0.810, 0.772,
        ])
        self.fallback_cdpp_duration_hr = 6.5

        self.detection_model = detection_model.lower().strip()
        self.sigmoid_steepness = float(sigmoid_steepness)

        self.use_observed_transit_flag_for_nasa = use_observed_transit_flag_for_nasa
        self.use_observed_transit_depth_for_nasa = use_observed_transit_depth_for_nasa
        self.assume_bright_if_kepmag_missing_for_nasa = assume_bright_if_kepmag_missing_for_nasa
        self.estimate_missing_semimajor_axis = estimate_missing_semimajor_axis

        self.catalog, self.source, self.nasa_like_source = data_processing.prepare_catalog(
            data,
            source,
            estimate_missing_semimajor_axis=self.estimate_missing_semimajor_axis,
            validate_for_detection=validate_for_detection,
        )

    # ============================================================
    # Flat-universe gate 1: does the planet transit?
    # ============================================================

    def _transiting(self):
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
        rs = pd.to_numeric(self.catalog["radius_s"], errors="coerce")
        k = pd.to_numeric(self.catalog.get("observed_radius_ratio", pd.Series(np.nan, index=self.catalog.index)), errors="coerce")
        observed = pd.to_numeric(self.catalog.get("observed_transit_depth_ppm", pd.Series(np.nan, index=self.catalog.index)), errors="coerce")
        model = pd.Series(transit_shape.radius_ratio(rp, rs), index=self.catalog.index)
        return k.fillna(model).fillna(np.sqrt(observed.clip(lower=0) / 1e6))

    def _a_over_rstar(self) -> pd.Series:
        a = pd.to_numeric(self.catalog["semimajor_p"], errors="coerce")
        rs = pd.to_numeric(self.catalog["radius_s"], errors="coerce")
        return pd.Series(transit_shape.a_over_rstar(a, rs), index=self.catalog.index)

    def _impact_from_inclination(self) -> pd.Series:
        inc_col = next((c for c in ["inc_p", "inclination", "pl_orbincl", "koi_incl"] if c in self.catalog.columns), None)
        if inc_col is None:
            return pd.Series(np.nan, index=self.catalog.index)
        inc = pd.to_numeric(self.catalog[inc_col], errors="coerce")
        return pd.Series(transit_shape.impact_from_inclination(self._a_over_rstar(), inc), index=self.catalog.index)

    def _impact_parameter(self) -> pd.Series:
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

    # ============================================================
    # Flat-universe gate 2: is the host bright enough?
    # ============================================================

    def _estimate_kepler_mag_from_bolometric_proxy(self):
        """Fallback only. Prefer real Kepler magnitude."""
        required = ["l_sun", "distance_s"]
        missing = [col for col in required if col not in self.catalog.columns]
        if missing:
            raise ValueError(
                f"Missing columns for brightness proxy: {missing}. Prefer using real kepmag."
            )

        l_sun = pd.to_numeric(self.catalog["l_sun"], errors="coerce").clip(lower=1e-12)
        distance_s = pd.to_numeric(self.catalog["distance_s"], errors="coerce").clip(lower=1e-12)
        return pd.Series(apparent_bolometric_mag(l_sun, distance_s), index=self.catalog.index)

    def _bright_enough_kepler(self):
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
            approx_mbol = self._estimate_kepler_mag_from_bolometric_proxy()
            bright = approx_mbol <= self.kepler_mag_limit
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
    # Flat-universe gate 3: enough transits and high enough MES
    # ============================================================

    def _transit_depth_fraction(self):
        return self._radius_ratio() ** 2

    # Calibration-only input A: real KOI/PSCompPars transit depths can replace the model depth.
    def _transit_depth_ppm(self):
        """Observed depth for calibration rows when available; otherwise model (Rp/R*)^2."""
        if self.nasa_like_source and self.use_observed_transit_depth_for_nasa:
            if "observed_transit_depth_ppm" in self.catalog.columns:
                observed = pd.to_numeric(self.catalog["observed_transit_depth_ppm"], errors="coerce")
                model = self._transit_depth_fraction() * 1e6
                depth = observed.fillna(model)
                self.catalog["transit_depth_source"] = np.where(observed.notna(), "observed", "model_Rp_Rstar")
                return depth
        self.catalog["transit_depth_source"] = "model_Rp_Rstar"
        return self._transit_depth_fraction() * 1e6

    def _signal_rms_ppm(self) -> pd.Series:
        """Limb-darkened transit RMS over T14; this is the signal term in MES."""
        observed = None
        if self.nasa_like_source and self.use_observed_transit_depth_for_nasa and "observed_transit_depth_ppm" in self.catalog.columns:
            observed = pd.to_numeric(self.catalog["observed_transit_depth_ppm"], errors="coerce")
        rms = transit_shape.signal_rms_ppm(self._radius_ratio(), self.catalog["impact_parameter"], "Kepler", observed)
        return pd.Series(rms, index=self.catalog.index)

    def _observed_duration(self) -> pd.Series:
        if self.nasa_like_source and "observed_transit_duration_hr" in self.catalog.columns:
            return pd.to_numeric(self.catalog["observed_transit_duration_hr"], errors="coerce")
        return pd.Series(np.nan, index=self.catalog.index)

    # Calibration-only input A, continued: real KOI/PSCompPars durations can replace model T14.
    def _transit_duration_hours(self):
        """T14 in hours: the observed value for NASA rows, else a circular orbit with impact parameter b."""
        observed = self._observed_duration()
        p_days = pd.to_numeric(self.catalog["p_orb"], errors="coerce")
        model = pd.Series(transit_shape.t14_hours(p_days, self._a_over_rstar(), self._radius_ratio(),
                                                  self.catalog["impact_parameter"]), index=self.catalog.index)
        duration_hr = observed.fillna(model).replace([np.inf, -np.inf], np.nan).fillna(6.0).clip(0.05, 48.0)
        self.catalog["transit_duration_hr_est"] = duration_hr
        self.catalog["transit_duration_source"] = np.where(observed.notna(), "observed", "model_t14")
        return duration_hr

    def _estimate_cdpp_from_kepler_mag(self):
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
            mag = self._estimate_kepler_mag_from_bolometric_proxy()
        else:
            mag = pd.Series(np.nan, index=self.catalog.index)

        cdpp_photon = self.fallback_cdpp_ppm * 10 ** (0.2 * (mag - self.cdpp_kp_ref_mag))
        cdpp_photon = cdpp_photon.replace([np.inf, -np.inf], np.nan).fillna(self.fallback_cdpp_ppm)

        # Add the stellar-variability floor in quadrature so bright F/G dwarfs (Kp ~7-9 at 60 pc) don't
        # become noiseless. The 10**(0.2 dKp) scaling is photon-limited, so it underestimates noise
        # for Kp >~ 14 where background and read noise matter; within 60 pc that is only M dwarfs.
        cdpp = np.sqrt(cdpp_photon ** 2 + self.cdpp_variability_ppm ** 2)
        return cdpp.clip(lower=self.cdpp_min_ppm, upper=self.cdpp_max_ppm)

    # Calibration-only input B: real Kepler rrmscdpp columns replace the magnitude fallback CDPP.
    def _cdpp_ppm_for_duration(self, duration_hr):
        """CDPP at T14: the star's own rrmscdpp columns interpolated log-log (extrapolated past 1.5
        or 15 h), else the magnitude fallback scaled from 6.5 h by the measured CDPP_SCALING."""
        duration_hr = np.asarray(duration_hr, float)
        grid = self.kepler_tps_durations_hr
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

        scale = (transit_shape.loglog_interp(duration_hr, self.cdpp_scaling_hr, self.cdpp_scaling)
                 / transit_shape.loglog_interp(self.fallback_cdpp_duration_hr, self.cdpp_scaling_hr, self.cdpp_scaling))
        fallback = self._estimate_cdpp_from_kepler_mag().to_numpy(float) * scale

        cdpp_ppm = pd.Series(np.where(has_own, own, fallback), index=self.catalog.index)
        self.catalog["kepler_cdpp_ppm"] = cdpp_ppm
        self.catalog["kepler_cdpp_source"] = np.where(
            has_own, "rrmscdpp_interpolated",
            "kepmag_cdpp_fallback" if self.use_kepmag_cdpp_fallback else "fallback_cdpp_ppm")
        return cdpp_ppm

    def _number_of_observed_transits(self):
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

    def _kepler_mes(self):
        """MES = rms dip over T14 / CDPP(T14) * sqrt(N transits)."""
        transit_depth_ppm = self._transit_depth_ppm()
        signal_ppm = self._signal_rms_ppm()
        duration_hr = self._transit_duration_hours()
        cdpp_ppm = self._cdpp_ppm_for_duration(duration_hr)
        n_transits = self._number_of_observed_transits()

        sqrt_n = np.sqrt(n_transits.clip(lower=1))
        mes = (signal_ppm / cdpp_ppm.replace(0, np.nan) * sqrt_n).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.catalog["transit_depth_ppm"] = transit_depth_ppm
        self.catalog["signal_rms_ppm"] = signal_ppm
        self.catalog["kepler_mes"] = mes
        self.catalog["kepler_mes_threshold"] = self.mes_threshold
        return mes

    def _mes_detection_pass(self):
        mes = self._kepler_mes()
        n_transits = self.catalog["n_transits_keplerish"]
        depth_good = (n_transits >= self.min_transits) & (mes >= self.mes_threshold)

        self.catalog["kepler_enough_transits"] = n_transits >= self.min_transits
        return depth_good.fillna(False)

    def _add_miss_reason_category(self):
        """Explain why each row passes or fails the current toy Kepler detector."""
        transiting = self.catalog["transiting_geometric"].astype(bool)
        bright = self.catalog["bright_enough_kepler"].astype(bool)
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
        Run the toy Kepler detector.

        Flat-universe detection uses three gates:
            1. transiting geometry / observed transit flag
            2. bright enough for Kepler follow-up
            3. enough transits and MES >= threshold

        Calibration rows can use observed depth, duration and CDPP, but the final logical
        detector is intentionally the same threshold model.
        """
        # 1. Transit geometry: P-Pop inclinations; NASA/KOI rows may use observed tran_flag.
        transiting = self._transiting()
        self._impact_parameter()

        # 2. Brightness gate: real Kp when available, otherwise a labeled bolometric proxy.
        bright_enough_kepler = self._bright_enough_kepler()

        # 3. Detection statistic: >= min_transits and MES >= mes_threshold.
        depth_good = self._mes_detection_pass()

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
        self.catalog["kepler_depth_pass"] = depth_good
        self.catalog["detected"] = detected

        # Kepler has one scenario; the shared plotters read best/worst (real for HWO and LIFEsim).
        self.catalog["detected_best"] = detected
        self.catalog["detected_worst"] = detected

        self._add_miss_reason_category()

        return self.catalog


# ============================================================
# Future upgrades: keep commented for later
# ============================================================

# FUTURE UPGRADE 1: replace CDPP * sqrt(N_transits) with the official DR25 one-sigma depth maps
# (KeplerPORTs: MES = depth_ppm / one_sigma_depth_ppm * 1.003), which include period-dependent gaps.
# Also not included: TPS vetting beyond threshold-crossing, odd/even checks, secondary eclipses,
# centroid vetting, injection-recovery completeness, dilution, and catalog disposition logic.

# UPGRADE 2 (implemented): detection_model='sigmoid' gives
# kepler_p_detect = logistic(sigmoid_steepness * (MES - mes_threshold)), used as a weight.
# Fuller option: KeplerPORTs DEMod.final_detEffs(MES, period).

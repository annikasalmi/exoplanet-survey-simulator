"""HARPS/NIRPS-style RV detector for P-Pop and NASA PSCompPars tables.

RADIAL VELOCITY MODEL

Included for flat-universe simulations:
- K semi-amplitude formula:
  K [m/s] = 28.4329 * m_jup * M_star^(-2/3) * P_year^(-1/3) / sqrt(1 - e^2)
- SNR threshold: detected when K / sigma_K >= 5.0 by default
- instrument model: HARPS (High Accuracy Radial velocity Planet Searcher) and
  NIRPS (Near InfraRed Planet Searcher), with photon noise, instrument floor,
  stellar-jitter RMS by spectral type, and n_obs = 100 observations by default
- eccentricity: clipped to [0, 0.95]
- M sin i preference: use pl_msinie if available; otherwise use pl_bmasse / mass_p

Included for calibration against real RV planets:
- published K values: pl_rvamp from NASA PSCompPars is normalized to rv_amp_obs
- ESO archive matching: plotting/paper_figures/recovery_3x1.py marks which hosts
  were actually observed by HARPS/NIRPS

Not included:
- long-term trends / polynomial drift, explicit observing cadence, full correlated
  stellar-activity models, multi-planet dynamics, or non-HARPS/NIRPS RV instruments
  such as CARMENES, ESPRESSO, APF, HIRES, etc.

Calibration data sources:
- NASA PSCompPars: published K, masses and orbital parameters
- ESO HARPS/NIRPS archive matching in the calibration plotting code
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from science import physics_constants as const
from science.physics import apparent_bolometric_mag
from science.telescopes.rv import data_processing


class RVData:
    TEFF_GRID = np.array([2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000], dtype=float)
    BC_GRIDS = {
        "V": np.array([-4.0, -2.6, -1.6, -1.0, -0.6, -0.35, -0.2, -0.1, -0.05, -0.1], dtype=float),
        "J": np.array([2.30, 2.10, 1.90, 1.70, 1.55, 1.45, 1.35, 1.25, 1.00, 0.70], dtype=float),
    }
    RV_K_PREFACTOR_MS = 28.4329

    # Per-instrument presets. NIRPS (J band) sees M dwarfs bright and less jittery, covering where
    # HARPS fails; take the per-planet best instrument, don't average. Constants from published
    # campaigns (e.g. HARPS-N TOI-1453: 100 points, 1.56 m/s). phot_full_mag = faintest mag that
    # still reaches sigma_phot_ref with longer exposures; fainter stars get noisier.
    INSTRUMENT_PRESETS = {
        # HARPS jitter grounded to Bellotti & Korhonen (2021) Table 3 (visible domain),
        # median peak-to-peak / 2.8, with A extrapolated.
        "HARPS": dict(band="V", sigma_instr_ms=0.8, sigma_phot_ref_ms=1.0, phot_full_mag=12.0,
                      jitter_by_stype={"A": 5.0, "F": 1.6, "G": 2.4, "K": 3.5, "M": 0.25, "Unknown": 2.0}),
        # NIRPS: NIR band J; M dwarfs bright + lower activity jitter in the IR.  Bellotti &
        # Korhonen (2021) is VISIBLE-domain (bluer = larger jitter), so their values do not
        # transfer to the NIR; NIRPS jitter kept lower here by design.  Not grounded to their table.
        "NIRPS": dict(band="J", sigma_instr_ms=1.0, sigma_phot_ref_ms=1.0, phot_full_mag=11.0,
                      jitter_by_stype={"A": 6.0, "F": 4.0, "G": 2.5, "K": 1.6, "M": 1.2, "Unknown": 2.0}),
    }

    def __init__(
        self,
        data: Union[pd.DataFrame, object],
        source: str = "auto",
        *,
        # --- instrument preset (HARPS default; NIRPS = NIR, for M dwarfs) ---
        instrument: str = "HARPS",
        band: Optional[str] = None,        # 'V' (optical) or 'J' (NIR); from preset if None
        # --- survey constants (None -> taken from the instrument preset) ---
        sigma_instr_ms: Optional[float] = None,   # instrumental/long-term RV floor (m/s)
        sigma_phot_ref_ms: Optional[float] = None,# achievable photon-noise floor (well-exposed)
        phot_full_mag: Optional[float] = None,    # faintest band-mag still reaching the photon floor
        # Let stars brighter than phot_full_mag go below the photon floor, down to sigma_phot_sys_floor,
        # so the brightest M dwarfs aren't under-credited (Proxima: 4.7 vs real ~20 sigma). Lifts easy-
        # planet recovery 79% -> 87%. True = default; False = legacy clamp.
        photon_beats_floor: bool = True,
        sigma_phot_sys_floor: float = 0.15,       # irreducible photon/systematic floor (m/s)
        jitter_by_stype: Optional[dict] = None,
        n_obs: int = 100,                  # RV epochs (HARPS-N small-planet campaigns: ~93-100)
        jitter_red_frac: float = 0.0,      # fraction of stellar-activity jitter treated as
                                           # correlated (red): an irreducible floor on K. Default 0: on 342 real
                                           # small planets f=0 recovers 57.6% at >=5 sigma (58.5% published),
                                           # f=0.5 only 10.8%. Kept as a sensitivity knob.
        snr_threshold: float = 5.0,        # 5-sigma "secure mass" (analog of 7.1 MES)
        vmag_limit: float = 16.0,          # generous faint cutoff for follow-up feasibility
        # --- mass / signal options ---
        apply_sini: bool = True,           # use Mp*sin i (realistic) vs true Mp
        # --- detection-efficiency model (matches kepler/tess) ---
        detection_model: str = "threshold",   # 'threshold' (hard) or 'sigmoid' (weight)
        sigmoid_steepness: float = 1.0,
        # --- stellar mass handling ---
        # mass_s in P-Pop catalogs is a constant placeholder (1 M_sun in kg), so we
        # derive M_star from radius on the lower main sequence (M/Msun ~ R/Rsun) unless
        # the supplied mass_s already looks like solar units (NASA st_mass does).
        stellar_mass_radius_exponent: float = 1.0,
        validate_for_detection: bool = True,
    ):
        # Resolve instrument preset; explicit kwargs (not None) override preset values.
        self.instrument = str(instrument).upper().strip()
        preset = self.INSTRUMENT_PRESETS.get(self.instrument, self.INSTRUMENT_PRESETS["HARPS"])
        self.band = (band or preset["band"]).upper().strip()

        # Teff -> bolometric correction (M_bol = M_band + BC_band, so M_band = M_bol - BC_band).
        # V makes cool stars faint for optical RV; J makes them bright for NIRPS.
        if self.band not in self.BC_GRIDS:
            raise ValueError(f"band must be one of {list(self.BC_GRIDS)}, got {self.band!r}")
        self.sigma_instr_ms = float(sigma_instr_ms if sigma_instr_ms is not None else preset["sigma_instr_ms"])
        self.sigma_phot_ref_ms = float(sigma_phot_ref_ms if sigma_phot_ref_ms is not None else preset["sigma_phot_ref_ms"])
        self.phot_full_mag = float(phot_full_mag if phot_full_mag is not None else preset["phot_full_mag"])
        self.photon_beats_floor = bool(photon_beats_floor)
        self.sigma_phot_sys_floor = float(sigma_phot_sys_floor)
        self.jitter_by_stype = dict(preset["jitter_by_stype"])
        if jitter_by_stype:
            self.jitter_by_stype.update(jitter_by_stype)
        self.n_obs = int(n_obs)
        self.jitter_red_frac = float(np.clip(jitter_red_frac, 0.0, 1.0))
        self.snr_threshold = float(snr_threshold)
        self.vmag_limit = float(vmag_limit)
        self.apply_sini = bool(apply_sini)
        self.detection_model = detection_model.lower().strip()
        self.sigmoid_steepness = float(sigmoid_steepness)
        self.stellar_mass_radius_exponent = float(stellar_mass_radius_exponent)

        self.catalog, self.source = data_processing.prepare_catalog(
            data,
            source,
            validate_for_detection=validate_for_detection,
        )

    # ------------------------------------------------------------------
    # Flat-universe input: stellar mass and instrumental brightness
    # ------------------------------------------------------------------

    def stellar_mass_msun(self) -> pd.Series:
        """Stellar mass in solar units. Uses mass_s only if already in solar units (NASA st_mass);
        P-Pop's mass_s (1 M_sun in kg) is replaced by a main-sequence estimate from radius.
        """
        n = len(self.catalog)
        src = pd.Series("radius_estimate", index=self.catalog.index, dtype=object)

        m_solar = pd.Series(np.nan, index=self.catalog.index)
        if "mass_s" in self.catalog.columns:
            ms = pd.to_numeric(self.catalog["mass_s"], errors="coerce")
            plausible = ms.between(0.05, 3.0)
            m_solar = m_solar.where(~plausible, ms)
            src = src.where(~plausible, "mass_s_catalog")

        if "radius_s" in self.catalog.columns:
            rs = pd.to_numeric(self.catalog["radius_s"], errors="coerce").clip(lower=0.05, upper=3.0)
            m_from_r = rs ** self.stellar_mass_radius_exponent
            need = m_solar.isna() & rs.notna()
            m_solar = m_solar.where(~need, m_from_r)

        # Last resort: 1 M_sun.
        fallback = m_solar.isna()
        m_solar = m_solar.where(~fallback, 1.0)
        src = src.where(~fallback, "assumed_1Msun")

        m_solar = m_solar.clip(lower=0.05, upper=3.0)
        self.catalog["rv_mass_s_msun"] = m_solar
        self.catalog["rv_mass_s_source"] = src
        return m_solar

    def _mag_from_lum(self, bc_grid: np.ndarray) -> pd.Series:
        """Apparent magnitude in a band from luminosity + distance + Teff-dependent BC."""
        if not {"l_sun", "distance_s"}.issubset(self.catalog.columns):
            return pd.Series(np.nan, index=self.catalog.index)
        l = pd.to_numeric(self.catalog["l_sun"], errors="coerce").clip(lower=1e-12)
        d = pd.to_numeric(self.catalog["distance_s"], errors="coerce").clip(lower=1e-6)
        m_bol = pd.Series(apparent_bolometric_mag(l, d), index=self.catalog.index)
        if "teff_s" in self.catalog.columns:
            teff = pd.to_numeric(self.catalog["teff_s"], errors="coerce")
            bc = pd.Series(np.interp(teff.clip(2500, 8000).to_numpy(float), self.TEFF_GRID, bc_grid),
                           index=self.catalog.index)
            return m_bol - bc  # M_bol = M_band + BC_band
        return m_bol

    def apparent_vmag(self) -> pd.Series:
        """Apparent V magnitude, from catalog vmag/gaiamag or else luminosity, distance and BC_V."""
        v = pd.Series(np.nan, index=self.catalog.index)
        src = pd.Series("missing", index=self.catalog.index, dtype=object)
        if "vmag" in self.catalog.columns:
            cv = pd.to_numeric(self.catalog["vmag"], errors="coerce")
            v = v.where(cv.isna(), cv); src = src.where(cv.isna(), "vmag_catalog")
        m_v = self._mag_from_lum(self.BC_GRIDS["V"])
        need = v.isna() & m_v.notna()
        v = v.where(~need, m_v); src = src.where(~need, "vmag_from_lum_distance_bc")
        if "gaiamag" in self.catalog.columns:
            g = pd.to_numeric(self.catalog["gaiamag"], errors="coerce")
            need = v.isna() & g.notna()
            v = v.where(~need, g); src = src.where(~need, "gaiamag_proxy")
        self.catalog["rv_vmag"] = v
        self.catalog["rv_vmag_source"] = src
        return v

    def apparent_band_mag(self) -> pd.Series:
        """Apparent magnitude in the instrument band (V for HARPS, J for NIRPS); drives photon noise
        and the brightness gate. Cool stars are bright in J, so NIRPS can reach M dwarfs.
        """
        v = self.apparent_vmag()  # ensures rv_vmag exists
        if self.band == "V":
            mag = v.copy(); src = self.catalog["rv_vmag_source"].copy()
        else:
            mag = self._mag_from_lum(self.BC_GRIDS[self.band])
            src = pd.Series(f"mag_{self.band}_from_lum_distance_bc", index=self.catalog.index, dtype=object)
            # fall back to V where the band magnitude could not be built
            mag = mag.where(mag.notna(), v)
        self.catalog["rv_band"] = self.band
        self.catalog["rv_mag"] = mag
        self.catalog["rv_mag_source"] = src
        return mag

    # ------------------------------------------------------------------
    # Flat-universe signal: planet mass choice and K semi-amplitude
    # ------------------------------------------------------------------

    def planet_mass_for_signal_mearth(self) -> pd.Series:
        """Mass that enters K: Mp*sin i (realistic) or true Mp, in Earth masses.

        For NASA rows the projected mass msini_p is used directly when present.
        """
        mass = None
        if "mass_p" in self.catalog.columns:
            mass = pd.to_numeric(self.catalog["mass_p"], errors="coerce")
        if "msini_p" in self.catalog.columns:
            msini = pd.to_numeric(self.catalog["msini_p"], errors="coerce")
            mass = msini if mass is None else mass.fillna(msini)

        if mass is None:
            raise ValueError("No planet mass column (mass_p / msini_p) available.")

        sini = pd.Series(1.0, index=self.catalog.index)
        if self.apply_sini and "inc_p" in self.catalog.columns:
            inc = pd.to_numeric(self.catalog["inc_p"], errors="coerce")
            # P-Pop inc_p is radians (0..pi); NASA pl_orbincl is degrees.
            inc_rad = inc if (inc.abs().max(skipna=True) or 0) <= 3.2 else np.deg2rad(inc)
            sini = np.sin(inc_rad).abs().clip(lower=0.0, upper=1.0)
            # If a projected mass (msini) was supplied, it already includes sin i.
            if "msini_p" in self.catalog.columns:
                used_msini = pd.to_numeric(self.catalog["msini_p"], errors="coerce").notna()
                sini = sini.where(~used_msini, 1.0)

        self.catalog["rv_sini_used"] = sini if self.apply_sini else 1.0
        return (mass * sini).rename("rv_signal_mass_mearth")

    def calc_semiamplitude(self) -> pd.Series:
        """RV semi-amplitude K in m/s (canonical formula)."""
        m_signal_mearth = self.planet_mass_for_signal_mearth()
        m_jup = m_signal_mearth * const.M_EARTH_IN_M_JUP
        mstar = self.stellar_mass_msun()
        p_yr = pd.to_numeric(self.catalog["p_orb"], errors="coerce") / 365.25
        ecc = pd.to_numeric(self.catalog["ecc_p"], errors="coerce").fillna(0.0).clip(0.0, 0.95)

        k = (
            self.RV_K_PREFACTOR_MS
            * m_jup
            * mstar ** (-2.0 / 3.0)
            * p_yr ** (-1.0 / 3.0)
            / np.sqrt(1.0 - ecc ** 2)
        )
        k = k.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        self.catalog["rv_k_ms"] = k
        return k

    # ------------------------------------------------------------------
    # Flat-universe noise model: instrument + photon noise + toy stellar jitter
    # ------------------------------------------------------------------

    def calc_noise(self) -> pd.Series:
        """Per-measurement RV precision sigma_rv (m/s): instrument, photon and jitter in quadrature.
        Photon noise uses the band magnitude, so NIRPS sees M dwarfs as bright and HARPS as faint.
        """
        mag = self.apparent_band_mag()
        # Adaptive exposure: photon noise stays at the floor down to phot_full_mag and grows as
        # 10^(0.2*excess) for fainter stars; brighter stars go below it, to sigma_phot_sys_floor.
        excess = mag - self.phot_full_mag
        if not self.photon_beats_floor:
            excess = excess.clip(lower=0.0)
        sigma_phot = (self.sigma_phot_ref_ms * 10 ** (0.2 * excess)).clip(lower=self.sigma_phot_sys_floor)
        sigma_phot = sigma_phot.replace([np.inf, -np.inf], np.nan).fillna(self.sigma_phot_ref_ms)

        jitter = self.catalog["stype"].map(self.jitter_by_stype).astype(float)
        jitter = jitter.fillna(self.jitter_by_stype.get("Unknown", 2.0))

        sigma_rv = np.sqrt(self.sigma_instr_ms ** 2 + sigma_phot ** 2 + jitter ** 2)
        self.catalog["rv_sigma_phot_ms"] = sigma_phot
        self.catalog["rv_sigma_jitter_ms"] = jitter
        self.catalog["rv_sigma_ms"] = sigma_rv
        return sigma_rv

    # ------------------------------------------------------------------
    # Flat-universe detection statistic: K / sigma_K
    # ------------------------------------------------------------------

    def calc_snr(self) -> pd.Series:
        """rv_snr = K / sigma_K, with sigma_K^2 = (2/N)[instr^2 + phot^2 + ((1-f) jit)^2] + (f jit)^2.
        f = jitter_red_frac is the correlated jitter share that doesn't average down; f = 0 gives the
        all-white sqrt(2/N_obs) * sigma_rv.
        """
        k = self.calc_semiamplitude()
        self.calc_noise()  # populates rv_sigma_ms / rv_sigma_phot_ms / rv_sigma_jitter_ms
        f = self.jitter_red_frac
        jit = pd.to_numeric(self.catalog["rv_sigma_jitter_ms"], errors="coerce").fillna(0.0)
        phot = pd.to_numeric(self.catalog["rv_sigma_phot_ms"], errors="coerce").fillna(0.0)
        white_var = self.sigma_instr_ms ** 2 + phot ** 2 + ((1.0 - f) * jit) ** 2
        red_floor_var = (f * jit) ** 2
        sigma_k = np.sqrt((2.0 / max(self.n_obs, 1)) * white_var + red_floor_var)
        snr = k / sigma_k.replace(0, np.nan)
        snr = snr.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self.catalog["rv_n_obs"] = self.n_obs
        self.catalog["rv_sigma_K_ms"] = sigma_k
        self.catalog["rv_snr"] = snr
        self.catalog["rv_snr_threshold"] = self.snr_threshold
        return snr

    def bright_enough(self) -> pd.Series:
        mag = pd.to_numeric(self.catalog["rv_mag"], errors="coerce") if "rv_mag" in self.catalog.columns else self.apparent_band_mag()
        bright = mag.le(self.vmag_limit).fillna(False)
        self.catalog["rv_vmag_limit"] = self.vmag_limit
        self.catalog["rv_star_bright_enough"] = bright
        return bright

    def classify_reasons(self) -> pd.Series:
        bright = self.catalog["rv_star_bright_enough"].astype(bool)
        snr = pd.to_numeric(self.catalog["rv_snr"], errors="coerce").fillna(0.0)
        jitter = pd.to_numeric(self.catalog["rv_sigma_jitter_ms"], errors="coerce").fillna(0.0)
        sigma = pd.to_numeric(self.catalog["rv_sigma_ms"], errors="coerce").fillna(np.inf)
        detected = self.catalog["rv_detected"].astype(bool)

        reason = np.full(len(self.catalog), "signal_too_weak", dtype=object)
        reason[(~bright).to_numpy()] = "host_star_too_faint"
        # among bright-but-undetected, flag the dominant noise term
        below = bright & (snr < self.snr_threshold)
        jitter_dom = below & (jitter >= sigma * 0.7)
        reason[(below & ~jitter_dom).to_numpy()] = "signal_too_weak"
        reason[jitter_dom.to_numpy()] = "jitter_dominated"
        reason[detected.to_numpy()] = "detected"

        self.catalog["rv_reason_category"] = reason
        self.catalog["reason_category"] = reason
        return self.catalog["rv_reason_category"]

    # ------------------------------------------------------------------
    # Final detector
    # ------------------------------------------------------------------

    def determine_detectable(self) -> pd.DataFrame:
        """Run the toy RV detector.

        Flat-universe detection uses two gates:
            1. target is bright enough for the chosen HARPS/NIRPS preset
            2. RV mass SNR = K / sigma_K passes snr_threshold

        Calibration rows can carry published pl_rvamp -> rv_amp_obs for comparison, but
        the detector always uses the same forward model for K and sigma_K.
        """
        # 1. Forward model the RV signal and its uncertainty.
        snr = self.calc_snr()

        # 2. Target feasibility: apparent magnitude in the active instrument band.
        bright = self.bright_enough()

        # 3. Detection threshold: default is a 5-sigma secure-mass cut.
        snr_pass = snr >= self.snr_threshold
        detected = bright & snr_pass
        self.catalog["rv_snr_pass"] = snr_pass
        self.catalog["rv_detected"] = detected

        # Detection probability weight — NOT a Bernoulli draw (mirrors kepler/tess).
        # Usage: expected_detected_in_bin = rv_p_detect[mask].sum()
        if self.detection_model == "sigmoid":
            p_snr = 1.0 / (1.0 + np.exp(-self.sigmoid_steepness * (snr - self.snr_threshold)))
        else:
            p_snr = snr_pass.astype(float)
        self.catalog["rv_p_detect"] = p_snr.where(bright, 0.0)

        # RV has one scenario; the shared plotters read best/worst (real for HWO and LIFEsim).
        self.catalog["detected"] = detected
        self.catalog["detected_best"] = detected
        self.catalog["detected_worst"] = detected

        self.classify_reasons()
        return self.catalog

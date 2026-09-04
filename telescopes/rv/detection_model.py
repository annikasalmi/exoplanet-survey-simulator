"""
rv_data.py — HARPS / HARPS-N-style radial-velocity detector for P-Pop + NASA.

Sibling of kepler_data.py / tess_data.py.  The transit detectors decide whether a
planet's *transit* is recoverable; this one decides whether a planet's *mass* is
recoverable by an optical RV follow-up campaign.  Same usage pattern:

    RVData(PPopObj.catalog, source="ppop").determine_detectable()
    RVData(nasa_df, source="pscomppars").determine_detectable()

Detection rule (hard threshold, analog of Kepler's 7.1 MES / TESS's 7.1 SNR):

    rv_detected = bright_enough AND (rv_snr >= snr_threshold)

with the semi-amplitude signal-to-noise definition

    sigma_K = sqrt(2 / N_obs) * sigma_rv     # uncertainty on K from N white-noise epochs
    rv_snr  = K / sigma_K = K * sqrt(N_obs / 2) / sigma_rv

    K       = RV semi-amplitude (m/s), from the planet mass            [signal]
    sigma_rv= per-measurement RV precision (m/s)                       [noise]
    N_obs   = number of RV epochs in the follow-up campaign            [looks]

(This is the standard sqrt(2/N) semi-amplitude error; it is a factor sqrt(2)
stricter than the bare K*sqrt(N)/sigma "looks" form.)

K (canonical form; the whiteboard's 29.8 m/s is a minor-constant variant):

    K = 28.4329 m/s * (Mp*sin i / M_Jup) * (M_star/M_sun)^(-2/3)
                    * (P / 1 yr)^(-1/3) / sqrt(1 - e^2)

Mass source for K (M sin i vs true mass):
    apply_sini=True  -> uses Mp * sin(inc)   (realistic: RV only ever measures M sin i)
    apply_sini=False -> uses Mp              (true mass; for the "what if i were known" run)
    Toggle it to see the selection effect of the unknown inclination.

Survey constants default to the HARPS / HARPS-N family — the dominant optical
spectrographs behind the measured masses of small Kepler/TESS planets.  HARPS and
HARPS-N are twin instruments (R~115,000), so a one- or two-survey model shares the
same constants.  All numbers are documented parameters; change them per facility.

This is a toy / survey-averaged model.  It does not use a real cadence, window
function, activity time series, or per-target RV jitter measurement.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

try:
    from lifesim.core.data import Data
except Exception:  # lets this file import outside the lifesim environment
    Data = None


class RVData:
    # ------------------------------------------------------------------
    # Physical constants
    # ------------------------------------------------------------------
    M_EARTH_IN_M_JUP = 1.0 / 317.828  # Earth masses -> Jupiter masses
    K_CONST_MS = 28.4329              # m/s, canonical RV semi-amplitude prefactor

    # Stellar (spot-induced) RV jitter (m/s RMS) by spectral type, VISIBLE domain.
    # Grounded to Bellotti & Korhonen (2021), Astron. Nachr. 342, 926, Table 3
    # (DOI 10.1002/asna.20210003): median mean peak-to-peak jitter per type over
    # their 15-star F-M sample, converted p2p -> RMS by /2.8 (phase-sampled
    # rotational modulation, RMS ~ p2p / 2*sqrt(2)).  Their sample deliberately
    # spans activity levels, so per-type values partly reflect the sample's
    # activity mix (active K stars, inactive M stars), NOT a universal type law;
    # jitter tracks logR'HK / filling factor / vsini, per the paper.  A is not in
    # their F-M sample -> extrapolated.  NOTE: this OVERRIDES the earlier values
    # (F=2.5,G=1.2,K=1.3,M=2.5) that were tuned to match published pl_rvamperr on
    # 342 real small planets; expect M-dwarf jitter to drop ~10x here.
    JITTER_BY_STYPE_MS = {"A": 5.0, "F": 1.6, "G": 2.4, "K": 3.5, "M": 0.25, "Unknown": 2.0}

    # Teff -> bolometric correction (M_bol = M_band + BC_band, so M_band = M_bol - BC_band).
    # BC_V is large and NEGATIVE for cool stars -> M dwarfs faint in V (optical RV struggles).
    # BC_J is POSITIVE and large for cool stars -> M dwarfs BRIGHT in J (NIR RV / NIRPS wins).
    _TEFF_GRID = np.array([2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000], dtype=float)
    _BC_V_GRID = np.array([-4.0, -2.6, -1.6, -1.0, -0.6, -0.35, -0.2, -0.1, -0.05, -0.1], dtype=float)
    _BC_J_GRID = np.array([2.30, 2.10, 1.90, 1.70, 1.55, 1.45, 1.35, 1.25, 1.00, 0.70], dtype=float)
    _BC_GRIDS = {"V": _BC_V_GRID, "J": _BC_J_GRID}

    # Per-instrument presets. NIRPS observes in the NIR (band J): M dwarfs are bright there
    # and their activity jitter is lower than in the optical, so NIRPS complements HARPS
    # exactly where HARPS fails (faint, active M dwarfs).  Model them separately and take
    # the per-planet best instrument; do NOT average their noise into one model.
    # Constants calibrated to published small-planet campaigns:
    #   HARPS-N TOI-1453 (Roy+2025): 100 RV points, median per-point sigma 1.56 m/s, masses 1-3 Me.
    #   K2-136c (Mayo+2023): 93 points, 1.6 m/s.  NIRPS: ~1 m/s goal, ~48-70 spectra for a 3-sigma mass.
    # phot_full_mag = faintest band-mag that still reaches sigma_phot_ref by adapting exposure time;
    # beyond it the photon noise degrades.  This replaces the old fixed-exposure photon model that
    # made every target fainter than the reference far too noisy.
    INSTRUMENT_PRESETS = {
        # HARPS jitter grounded to Bellotti & Korhonen (2021) Table 3 (visible domain);
        # see JITTER_BY_STYPE_MS above for the p2p->RMS conversion and caveats.
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
        # Let stars BRIGHTER than phot_full_mag integrate BELOW the nominal photon floor,
        # down to sigma_phot_sys_floor.  Without this the noise is too FLAT vs brightness and
        # the model under-credits the brightest nearby M dwarfs (Proxima J=5.2 was scored 4.7
        # sigma vs its real ~20 sigma).  Validated in scripts/59: this lifts genuinely-easy
        # (real >=5 sigma) planet recovery 79%->87% while barely touching the marginal ones
        # (real <5 sigma: 57%->59%).  True = close-to-reality default; False = legacy clamp.
        photon_beats_floor: bool = True,
        sigma_phot_sys_floor: float = 0.15,       # irreducible photon/systematic floor (m/s)
        jitter_by_stype: Optional[dict] = None,
        n_obs: int = 100,                  # RV epochs (HARPS-N small-planet campaigns: ~93-100)
        jitter_red_frac: float = 0.0,      # fraction of stellar-activity jitter treated as
                                           # CORRELATED (red) -> irreducible floor on K that
                                           # does NOT average down with N epochs.
                                           # CALIBRATION (2026-06-16): against 342 real small
                                           # (R<=2.2) planets with published pl_rvamp, f=0 gives
                                           # model median sigma_K=0.27 m/s (published 0.31) and
                                           # recovers 57.6% at >=5 sigma -- matching the 58.5% that
                                           # were actually published at >=5 sigma. f>0 double-counts
                                           # activity (real teams GP-model it out) and rejects real
                                           # detections (f=0.5 -> only 10.8% recovered). So default 0.
                                           # Kept as a knob for sensitivity tests only.
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
        self.catalog = self._as_dataframe(data)
        self.source = self._infer_source(source)

        # Resolve instrument preset; explicit kwargs (not None) override preset values.
        self.instrument = str(instrument).upper().strip()
        preset = self.INSTRUMENT_PRESETS.get(self.instrument, self.INSTRUMENT_PRESETS["HARPS"])
        self.band = (band or preset["band"]).upper().strip()
        if self.band not in self._BC_GRIDS:
            raise ValueError(f"band must be one of {list(self._BC_GRIDS)}, got {self.band!r}")
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

        self.catalog = self._standardize(self.catalog, self.source)
        self._add_basic_columns()
        if validate_for_detection:
            self._validate()

    # ------------------------------------------------------------------
    # Setup / standardization
    # ------------------------------------------------------------------

    @staticmethod
    def _as_dataframe(data) -> pd.DataFrame:
        if Data is not None and isinstance(data, Data):
            return pd.DataFrame(data.catalog).copy()
        if isinstance(data, pd.DataFrame):
            return data.copy()
        return pd.DataFrame(data).copy()

    @staticmethod
    def _num(s) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    def _infer_source(self, source: str) -> str:
        source = str(source).lower().strip()
        if source != "auto":
            return source
        cols = set(self.catalog.columns)
        if {"pl_name", "pl_bmasse"}.issubset(cols) or {"pl_name", "pl_msinie"}.issubset(cols):
            return "pscomppars"
        if {"kepoi_name", "koi_prad"}.issubset(cols):
            return "koi"
        return "ppop"

    def _standardize(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        df = df.copy()
        rename = {
            # NASA PSCompPars
            "pl_name": "planet_name", "hostname": "host_name",
            "pl_orbper": "p_orb", "pl_orbsmax": "semimajor_p", "pl_orbincl": "inc_p",
            "pl_orbeccen": "ecc_p", "pl_rade": "radius_p",
            "pl_bmasse": "mass_p", "pl_msinie": "msini_p",
            "pl_rvamp": "rv_amp_obs", "pl_insol": "flux_p",
            "st_rad": "radius_s", "st_mass": "mass_s", "st_teff": "teff_s",
            "st_lum": "st_lum_log10", "sy_dist": "distance_s",
            "sy_vmag": "vmag", "sy_gaiamag": "gaiamag",
            # KOI-ish
            "kepoi_name": "planet_name", "koi_period": "p_orb", "koi_prad": "radius_p",
            "koi_incl": "inc_p", "koi_srad": "radius_s", "koi_smass": "mass_s",
            "koi_steff": "teff_s", "koi_insol": "flux_p",
            # P-Pop aliases
            "luminosity_s": "l_sun", "temp_s": "teff_s", "insolation": "flux_p",
            "Vmag": "vmag", "gaia_g_mag": "gaiamag",
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

        for col in ["mass_p", "msini_p", "radius_p", "p_orb", "semimajor_p", "inc_p",
                    "ecc_p", "radius_s", "mass_s", "teff_s", "l_sun", "distance_s",
                    "flux_p", "vmag", "gaiamag", "rv_amp_obs"]:
            if col in df.columns:
                df[col] = self._num(df[col])

        df["dataset_source"] = {
            "pscomppars": "NASA_PSCompPars", "nasa": "NASA_PSCompPars", "koi": "NASA_KOI",
        }.get(source, "P-Pop_simulated")
        return df

    def _add_basic_columns(self) -> None:
        if "stype" not in self.catalog.columns:
            teff = self.catalog.get("teff_s", pd.Series(np.nan, index=self.catalog.index))
            self.catalog["stype"] = teff.apply(self._stellar_type)
        else:
            # normalize to single uppercase letter
            self.catalog["stype"] = (
                self.catalog["stype"].astype(str).str.strip().str.upper().str[0]
                .where(lambda s: s.isin(["A", "F", "G", "K", "M"]), "Unknown")
            )
        if "ecc_p" not in self.catalog.columns:
            self.catalog["ecc_p"] = 0.0
        self.catalog["ecc_p"] = self._num(self.catalog["ecc_p"]).fillna(0.0).clip(0.0, 0.99)
        if "habitable" not in self.catalog.columns and "flux_p" in self.catalog.columns:
            flux = self._num(self.catalog["flux_p"])
            self.catalog["habitable"] = (flux >= 0.25) & (flux <= 2.0)

    @staticmethod
    def _stellar_type(teff) -> str:
        if pd.isna(teff):
            return "Unknown"
        teff = float(teff)
        if teff >= 7500: return "A"
        if teff >= 6000: return "F"
        if teff >= 5200: return "G"
        if teff >= 3700: return "K"
        if teff > 0: return "M"
        return "Unknown"

    def _validate(self) -> None:
        required = ["mass_p", "p_orb"]
        missing = [c for c in required if c not in self.catalog.columns]
        # mass_p may be absent for NASA rows that only carry msini_p
        if "mass_p" in missing and "msini_p" in self.catalog.columns:
            missing.remove("mass_p")
        if missing:
            raise ValueError(
                f"Missing required RV detection columns: {missing}. "
                "Need a planet mass (mass_p or msini_p) and orbital period (p_orb)."
            )

    # ------------------------------------------------------------------
    # Stellar properties
    # ------------------------------------------------------------------

    def stellar_mass_msun(self) -> pd.Series:
        """Stellar mass in solar units.

        Prefer a supplied mass_s only when it is already in plausible solar units
        (NASA st_mass).  P-Pop's mass_s is a constant 1 M_sun expressed in kg, so it
        is detected as non-solar and replaced by a main-sequence radius estimate
        (M/Msun ~ (R/Rsun)^exponent, exponent ~1 across the lower MS where M/K dwarfs live).
        """
        n = len(self.catalog)
        src = pd.Series("radius_estimate", index=self.catalog.index, dtype=object)

        m_solar = pd.Series(np.nan, index=self.catalog.index)
        if "mass_s" in self.catalog.columns:
            ms = self._num(self.catalog["mass_s"])
            plausible = ms.between(0.05, 3.0)
            m_solar = m_solar.where(~plausible, ms)
            src = src.where(~plausible, "mass_s_catalog")

        if "radius_s" in self.catalog.columns:
            rs = self._num(self.catalog["radius_s"]).clip(lower=0.05, upper=3.0)
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
        l = self._num(self.catalog["l_sun"]).clip(lower=1e-12)
        d = self._num(self.catalog["distance_s"]).clip(lower=1e-6)
        m_bol = 4.74 - 2.5 * np.log10(l) + 5 * np.log10(d / 10.0)
        if "teff_s" in self.catalog.columns:
            teff = self._num(self.catalog["teff_s"])
            bc = pd.Series(np.interp(teff.clip(2500, 8000).to_numpy(float), self._TEFF_GRID, bc_grid),
                           index=self.catalog.index)
            return m_bol - bc  # M_bol = M_band + BC_band
        return m_bol

    def apparent_vmag(self) -> pd.Series:
        """Apparent V magnitude (always computed, for cross-instrument target cuts/labels).

        Uses a catalog vmag/gaiamag when present, else builds it from luminosity and
        distance with BC_V so cool stars are correctly faint in V.
        """
        v = pd.Series(np.nan, index=self.catalog.index)
        src = pd.Series("missing", index=self.catalog.index, dtype=object)
        if "vmag" in self.catalog.columns:
            cv = self._num(self.catalog["vmag"])
            v = v.where(cv.isna(), cv); src = src.where(cv.isna(), "vmag_catalog")
        m_v = self._mag_from_lum(self._BC_V_GRID)
        need = v.isna() & m_v.notna()
        v = v.where(~need, m_v); src = src.where(~need, "vmag_from_lum_distance_bc")
        if "gaiamag" in self.catalog.columns:
            g = self._num(self.catalog["gaiamag"])
            need = v.isna() & g.notna()
            v = v.where(~need, g); src = src.where(~need, "gaiamag_proxy")
        self.catalog["rv_vmag"] = v
        self.catalog["rv_vmag_source"] = src
        return v

    def apparent_band_mag(self) -> pd.Series:
        """Apparent magnitude in the instrument's band (V for HARPS, J for NIRPS).

        This is the magnitude that drives RV photon noise and the brightness gate.
        In J, cool stars are bright (BC_J > 0), so M dwarfs become RV-reachable for NIRPS.
        """
        v = self.apparent_vmag()  # ensures rv_vmag exists
        if self.band == "V":
            mag = v.copy(); src = self.catalog["rv_vmag_source"].copy()
        else:
            mag = self._mag_from_lum(self._BC_GRIDS[self.band])
            src = pd.Series(f"mag_{self.band}_from_lum_distance_bc", index=self.catalog.index, dtype=object)
            # fall back to V where the band magnitude could not be built
            mag = mag.where(mag.notna(), v)
        self.catalog["rv_band"] = self.band
        self.catalog["rv_mag"] = mag
        self.catalog["rv_mag_source"] = src
        return mag

    # ------------------------------------------------------------------
    # Signal, noise, S/N
    # ------------------------------------------------------------------

    def planet_mass_for_signal_mearth(self) -> pd.Series:
        """Mass that enters K: Mp*sin i (realistic) or true Mp, in Earth masses.

        For NASA rows the projected mass msini_p is used directly when present.
        """
        mass = None
        if "mass_p" in self.catalog.columns:
            mass = self._num(self.catalog["mass_p"])
        if "msini_p" in self.catalog.columns:
            msini = self._num(self.catalog["msini_p"])
            mass = msini if mass is None else mass.fillna(msini)

        if mass is None:
            raise ValueError("No planet mass column (mass_p / msini_p) available.")

        sini = pd.Series(1.0, index=self.catalog.index)
        if self.apply_sini and "inc_p" in self.catalog.columns:
            inc = self._num(self.catalog["inc_p"])
            # P-Pop inc_p is radians (0..pi); NASA pl_orbincl is degrees.
            inc_rad = inc if (inc.abs().max(skipna=True) or 0) <= 3.2 else np.deg2rad(inc)
            sini = np.sin(inc_rad).abs().clip(lower=0.0, upper=1.0)
            # If a projected mass (msini) was supplied, it already includes sin i.
            if "msini_p" in self.catalog.columns:
                used_msini = self._num(self.catalog["msini_p"]).notna()
                sini = sini.where(~used_msini, 1.0)

        self.catalog["rv_sini_used"] = sini if self.apply_sini else 1.0
        return (mass * sini).rename("rv_signal_mass_mearth")

    def calc_semiamplitude(self) -> pd.Series:
        """RV semi-amplitude K in m/s (canonical formula)."""
        m_signal_mearth = self.planet_mass_for_signal_mearth()
        m_jup = m_signal_mearth * self.M_EARTH_IN_M_JUP
        mstar = self.stellar_mass_msun()
        p_yr = self._num(self.catalog["p_orb"]) / 365.25
        ecc = self._num(self.catalog["ecc_p"]).fillna(0.0).clip(0.0, 0.99)

        k = (
            self.K_CONST_MS
            * m_jup
            * mstar ** (-2.0 / 3.0)
            * p_yr ** (-1.0 / 3.0)
            / np.sqrt(1.0 - ecc ** 2)
        )
        k = k.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        self.catalog["rv_k_ms"] = k
        return k

    def calc_noise(self) -> pd.Series:
        """Per-measurement RV precision sigma_rv (m/s): instrument (+) photon (+) jitter.

        Photon noise uses the instrument's band magnitude, so NIRPS (band J) sees M dwarfs
        as bright while HARPS (band V) sees them as faint.
        """
        mag = self.apparent_band_mag()
        # Adaptive exposure: surveys integrate to the photon floor for any target brighter than
        # phot_full_mag, so sigma_phot stays at the floor up to that limit and only degrades
        # (~10^(0.2*excess)) for fainter stars. (excess = max(0, mag - phot_full_mag).)
        # Stars BRIGHTER than phot_full_mag (negative excess) integrate BELOW the floor, down to
        # an irreducible systematic floor sigma_phot_sys_floor -- lets the brightest nearby M dwarfs
        # reach their real sub-0.1 m/s precision instead of being clamped at ~1 m/s.
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

    def calc_snr(self) -> pd.Series:
        """rv_snr = K / sigma_K.

        White noise (instrument + photon) averages down with the number of
        epochs as sqrt(2/N_obs).  Stellar-activity jitter, however, is partly
        CORRELATED (red): a fraction `jitter_red_frac` does NOT beat down with
        more epochs and sets an irreducible activity floor on the recoverable
        semi-amplitude.  So

            sigma_K^2 = (2/N) * [sigma_instr^2 + sigma_phot^2 + ((1-f)*sigma_jit)^2]
                        + (f * sigma_jit)^2

        With f = jitter_red_frac = 0 this collapses to the optimistic all-white
        sqrt(2/N_obs)*sigma_rv; f ~ 0.5 is a realistic, more conservative model.
        """
        k = self.calc_semiamplitude()
        self.calc_noise()  # populates rv_sigma_ms / rv_sigma_phot_ms / rv_sigma_jitter_ms
        f = self.jitter_red_frac
        jit = self._num(self.catalog["rv_sigma_jitter_ms"]).fillna(0.0)
        phot = self._num(self.catalog["rv_sigma_phot_ms"]).fillna(0.0)
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
        mag = self._num(self.catalog["rv_mag"]) if "rv_mag" in self.catalog.columns else self.apparent_band_mag()
        bright = mag.le(self.vmag_limit).fillna(False)
        self.catalog["rv_vmag_limit"] = self.vmag_limit
        self.catalog["rv_star_bright_enough"] = bright
        return bright

    def classify_reasons(self) -> pd.Series:
        bright = self.catalog["rv_star_bright_enough"].astype(bool)
        snr = self._num(self.catalog["rv_snr"]).fillna(0.0)
        k = self._num(self.catalog["rv_k_ms"]).fillna(0.0)
        jitter = self._num(self.catalog["rv_sigma_jitter_ms"]).fillna(0.0)
        sigma = self._num(self.catalog["rv_sigma_ms"]).fillna(np.inf)
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
        self.catalog["miss_reason"] = reason
        return self.catalog["rv_reason_category"]

    # ------------------------------------------------------------------
    # Final detector
    # ------------------------------------------------------------------

    def determine_detectable(self) -> pd.DataFrame:
        self._validate()
        snr = self.calc_snr()
        bright = self.bright_enough()

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

        # Backward-compatible names so the shared plotting helpers can treat RV like a survey.
        self.catalog["detected"] = detected
        self.catalog["detected_best"] = detected
        self.catalog["detected_worst"] = detected

        self.classify_reasons()
        return self.catalog

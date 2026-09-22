"""TESS toy transit detector for P-Pop, NASA PSCompPars, KOI and TOI-style tables.

TESS MODEL

Included for flat-universe simulations:
- min transits: >= 3 observed transits required to pass detection
- SNR threshold: >= 7.1 (Sullivan et al. 2015, ApJ 809, 77), theoretically the
  same as Kepler; practical SPOC thresholds can be closer to 7.3 after cosmic-ray rejection
- transit shape: limb-darkened TESS-band signal RMS from Claret 2017, reducing
  effective depth by about 13%

Included for calibration against real TESS planets only, e.g. recovery_3x1.py:
- SPOC TCE MES: tce_max_mult_ev from multi-sector TCE tables validates model accuracy
- per-TIC per-sector CDPP: SPOC noise measurements for 2-minute cadence targets;
  used to compute SNR = depth / CDPP
- ExoFOP TOI sectors: catalog sectors drive the searched observing windows

Not included:
- FFI Full-Frame Image searches, including 30-minute cadence stars not in the
  pre-selected 2-minute target list
- DV Data Validation vetting after threshold crossing; the model stops at the TPS threshold
- QLP Quicklook Pipeline detections or dispositions

Calibration data sources:
- ExoFOP TOI table: 1,343 TOIs in the calibration sample
- SPOC TCE tables, carrying tce_max_mult_ev for each multi-sector run
- SPOC CDPP tables, carrying per-star per-sector noise floors
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from science.telescopes import transit_shape
from science.telescopes.tess import data_processing
from tools.paths import TESS_REFERENCE_DATA_DIR

class TESSData:
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

    # Proxy-Tmag colour terms vs Teff: G - T and mbol - T (subtract from G or mbol to get T).
    # Medians over 4,078 dwarf planet hosts (log g >= 4) in NASA PSCompPars, whose sy_tmag is
    # TIC-8 (Stassun et al. 2019); the 2500 K bin has only 4 stars. Re-derive with:
    # Re-derive with data_processing.print_tmag_corrections(TESSData.TEFF_GRID).
    TEFF_GRID = np.array([2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000, 8000], dtype=float)
    G_MINUS_T = np.array([1.79, 1.37, 1.16, 0.94, 0.71, 0.60, 0.50, 0.43, 0.32, 0.12], dtype=float)
    MBOL_MINUS_T = np.array([-0.30, 0.19, 0.35, 0.42, 0.43, 0.46, 0.44, 0.39, 0.29, 0.13], dtype=float)

    # Priority order of noise sources; each row reports the best one any of its sectors used.
    NOISE_SOURCES = [
        "official_spoc_cdpp_tic_sector",
        "official_spoc_cdpp_tic_any_sector",
        "spoc_cdpp_tmag_sector_bin",
        "spoc_cdpp_tmag_all_sectors",
        "smooth_tmag_fallback",
    ]
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
        min_transits: int = 3,
        snr_threshold: float = 7.1,
        tmag_limit: Optional[float] = None,
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
        # Optional binned SPOC CDPP reference table; not loaded by default for flat simulations.
        noise_table_path: Optional[Union[str, Path]] = None,
        use_mast_tic: bool = False,
        mast_max_rows: int = 500,
        # Per-TIC SPOC CDPP CSVs from MAST (results/catalogs/tess/CDPP); calibration-only.
        cdpp_dir: Optional[Union[str, Path]] = None,
        use_cdpp_tables: bool = False,
        # Last-resort noise when no SPOC table covers a row: 1-hr noise ~ ticgen's pre-launch model
        # (Sullivan et al. 2015 with its 60 ppm/hr systematic floor), within 10% for Tmag 6-12.
        smooth_noise_ref_ppm_1hr: float = 200.0,
        smooth_noise_floor_ppm_1hr: float = 60.0,
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
        self.sector_days = float(sector_days)
        self.dutycycle = float(dutycycle)
        self.default_n_sectors = int(default_n_sectors)
        self.min_transits = int(min_transits)
        self.snr_threshold = float(snr_threshold)
        self.tmag_limit = float(tmag_limit) if tmag_limit is not None else None
        self.phase_mode = phase_mode.lower().strip()
        self.rng = np.random.default_rng(random_seed)

        self.use_tesspoint = bool(use_tesspoint)
        self.condition_on_observed = bool(condition_on_observed)
        self.use_catalog_sectors = bool(use_catalog_sectors)
        self.sector_grid_path = Path(sector_grid_path) if sector_grid_path else TESS_REFERENCE_DATA_DIR / "sector_grid.npz"
        self.noise_table_path = Path(noise_table_path) if noise_table_path else None
        self.use_mast_tic = bool(use_mast_tic)
        self.mast_max_rows = int(mast_max_rows)
        self.cdpp_dir = Path(cdpp_dir) if cdpp_dir is not None else None
        self.use_cdpp_tables = bool(use_cdpp_tables)
        self.smooth_noise_ref_ppm_1hr = float(smooth_noise_ref_ppm_1hr)
        self.smooth_noise_floor_ppm_1hr = float(smooth_noise_floor_ppm_1hr)

        # upgrade parameters
        self.apply_mdwarf_tmag_correction = bool(apply_mdwarf_tmag_correction)
        self.cvz_ecliptic_lat_deg = float(cvz_ecliptic_lat_deg)
        self.cvz_n_sectors = int(cvz_n_sectors) if cvz_n_sectors is not None else None
        self.ffi_cadence_min = float(ffi_cadence_min) if ffi_cadence_min is not None else None
        self.detection_model = detection_model.lower().strip()
        self.sigmoid_steepness = float(sigmoid_steepness)

        self.catalog, self.source = data_processing.prepare_catalog(
            data,
            source,
            apply_mdwarf_tmag_correction=self.apply_mdwarf_tmag_correction,
            teff_grid=self.TEFF_GRID,
            g_minus_t=self.G_MINUS_T,
            mbol_minus_t=self.MBOL_MINUS_T,
            validate_for_detection=validate_for_detection,
        )
        self.cdpp_table = (
            data_processing.load_cdpp_tables(self.cdpp_dir, self.CDPP_COLS)
            if self.use_cdpp_tables else pd.DataFrame()
        )
        self.noise_table = (
            pd.read_csv(self.noise_table_path)
            if self.noise_table_path is not None and self.noise_table_path.exists()
            else pd.DataFrame()
        )

        if self.use_mast_tic:
            self.catalog = data_processing.enrich_from_tic(self.catalog, self.mast_max_rows)
        self.catalog, self._pairs = data_processing.prepare_visibility(
            self.catalog,
            use_catalog_sectors=self.use_catalog_sectors,
            use_tesspoint=self.use_tesspoint,
            sector_grid_path=self.sector_grid_path,
            condition_on_observed=self.condition_on_observed,
            default_n_sectors=self.default_n_sectors,
            cvz_n_sectors=self.cvz_n_sectors,
            cvz_ecliptic_lat_deg=self.cvz_ecliptic_lat_deg,
            sector_days=self.sector_days,
            dutycycle=self.dutycycle,
        )

    # ------------------------------------------------------------------
    # CDPP tables and noise
    # ------------------------------------------------------------------

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
    # Flat-universe gate 1: observed by TESS
    # ------------------------------------------------------------------

    # Visibility is set during initialization by one of:
    # - fixed sectors for flat simulations
    # - real tess-point sector grid
    # - catalog sectors for TOI calibration

    # ------------------------------------------------------------------
    # Flat-universe gate 2: does the planet transit?
    # ------------------------------------------------------------------

    def _transiting(self) -> pd.Series:
        """True if the planet crosses the stellar disk; NASA transiting catalogs may use tran_flag."""
        b = self._impact_from_inclination()
        self.catalog["tess_impact_parameter"] = b
        if self.source in {"pscomppars", "nasa", "koi"} and "tran_flag" in self.catalog.columns:
            self.catalog["tess_transiting_source"] = "tran_flag"
            return pd.to_numeric(self.catalog["tran_flag"], errors="coerce").fillna(0).astype(int).eq(1)
        if "inc_p" not in self.catalog.columns:
            raise ValueError("Need inc_p for geometric transits, or tran_flag for observed NASA transiting planets.")
        self.catalog["tess_transiting_source"] = "inclination_geometry"
        return (b <= 1 + self._radius_ratio()).fillna(False)

    def _radius_ratio(self) -> pd.Series:
        """k = Rp / R*, from the radii, else from a measured depth."""
        rp = pd.to_numeric(self.catalog["radius_p"], errors="coerce")
        rs = pd.to_numeric(self.catalog["radius_s"], errors="coerce")
        observed = pd.to_numeric(self.catalog.get("observed_depth_ppm", pd.Series(np.nan, index=self.catalog.index)), errors="coerce")
        model = pd.Series(transit_shape.radius_ratio(rp, rs), index=self.catalog.index)
        return model.fillna(np.sqrt(observed.clip(lower=0) / 1e6))

    def _a_over_rstar(self) -> pd.Series:
        a = pd.to_numeric(self.catalog["semimajor_p"], errors="coerce")
        rs = pd.to_numeric(self.catalog["radius_s"], errors="coerce")
        return pd.Series(transit_shape.a_over_rstar(a, rs), index=self.catalog.index)

    def _impact_from_inclination(self) -> pd.Series:
        if "inc_p" not in self.catalog.columns:
            return pd.Series(np.nan, index=self.catalog.index)
        inc = pd.to_numeric(self.catalog["inc_p"], errors="coerce")
        return pd.Series(transit_shape.impact_from_inclination(self._a_over_rstar(), inc), index=self.catalog.index)

    def _impact_parameter(self) -> pd.Series:
        """b from a measured T14, else the inclination, else (1 + k) / 2, the mean for isotropic
        orbits that transit. Catalogued inclinations come from fits with their own a/R*, so pairing
        them with the stellar a/R* here can put an observed transit off the star."""
        k = self._radius_ratio()
        b = pd.to_numeric(self.catalog["tess_impact_parameter"], errors="coerce")
        observed = pd.to_numeric(self.catalog.get("observed_duration_hr", pd.Series(np.nan, index=self.catalog.index)), errors="coerce")
        p = pd.to_numeric(self.catalog["p_orb"], errors="coerce")
        from_duration = pd.Series(transit_shape.impact_from_t14(observed, p, self._a_over_rstar(), k),
                                  index=self.catalog.index).where(observed.notna())
        source = np.where(from_duration.notna(), "observed_duration", np.where(b.notna(), "inclination", "mean_for_transiting"))
        b = from_duration.fillna(b).fillna((1 + k) / 2)
        self.catalog["tess_impact_parameter"] = b
        self.catalog["tess_impact_parameter_source"] = source
        return b

    # ------------------------------------------------------------------
    # Flat-universe gates 3 and 5: enough transits, then high enough SNR
    # ------------------------------------------------------------------

    # Calibration-only input A: observed TOI/KOI depths can replace the model depth.
    def _depth_ppm(self) -> pd.Series:
        """Mid-transit depth in ppm: the observed depth for NASA rows when present, else (Rp/R*)^2."""
        model = self._radius_ratio() ** 2 * 1e6
        observed = pd.to_numeric(self.catalog.get("observed_depth_ppm", pd.Series(np.nan, index=self.catalog.index)), errors="coerce")
        self.catalog["tess_transit_depth_source"] = np.where(observed.notna(), "observed", "model_Rp_Rstar")
        return observed.fillna(model)

    def _signal_rms_ppm(self) -> pd.Series:
        """rms of the limb-darkened transit dip over T14 (see transit_shape); what the SNR uses."""
        observed = pd.to_numeric(self.catalog.get("observed_depth_ppm", pd.Series(np.nan, index=self.catalog.index)), errors="coerce")
        rms = transit_shape.signal_rms_ppm(self._radius_ratio(), self.catalog["tess_impact_parameter"],
                                           "TESS", observed)
        return pd.Series(rms, index=self.catalog.index)

    # Calibration-only input A, continued: observed durations can replace model T14.
    def _duration_hr(self) -> pd.Series:
        """T14 in hours: the observed value when available, else a circular orbit with impact parameter b."""
        observed = pd.to_numeric(self.catalog.get("observed_duration_hr", pd.Series(np.nan, index=self.catalog.index)), errors="coerce")
        p = pd.to_numeric(self.catalog["p_orb"], errors="coerce")
        model = pd.Series(transit_shape.t14_hours(p, self._a_over_rstar(), self._radius_ratio(),
                                                  self.catalog["tess_impact_parameter"]), index=self.catalog.index)
        dur = observed.fillna(model).replace([np.inf, -np.inf], np.nan).fillna(3.0).clip(0.05, 48.0)
        self.catalog["tess_transit_duration_source"] = np.where(observed.notna(), "observed", "model_t14")
        return dur

    def _transit_counts(self) -> pd.DataFrame:
        """Transits in each observed window, one row per (catalog row, sector or continuous span).
        A window lasts span * sector_days * dutycycle days, like the Kepler detector's
        dataspan * dutycycle. phase_mode 'random' draws one phase per window; 'expected' uses the
        mean count d / P, which is fractional, so a planet with one transit every other sector counts.
        """
        pairs = self._pairs.copy()
        p = pd.to_numeric(self.catalog["p_orb"], errors="coerce").to_numpy(float)[pairs["row"].to_numpy()]
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

    def _snr(self, pairs: Optional[pd.DataFrame] = None) -> pd.Series:
        """TESS transit SNR over all observed sectors: SNR = rms dip * sqrt(sum_i N_i / CDPP_i^2), with
        CDPP interpolated log-log to T14. tess_cdpp_ppm is the effective per-transit CDPP."""
        if pairs is None:
            pairs = self._transit_counts()
        n_rows = len(self.catalog)
        signal = self._signal_rms_ppm().to_numpy(float)
        dur = self._duration_hr().to_numpy(float)
        tmag = pd.to_numeric(self.catalog["tess_tmag"], errors="coerce").to_numpy(float)
        tic = pd.to_numeric(self.catalog.get("ticid", pd.Series(np.nan, index=self.catalog.index)), errors="coerce").to_numpy(float)
        dilution = pd.to_numeric(self.catalog.get("tess_dilution", self.catalog.get("dilution", pd.Series(1.0, index=self.catalog.index))), errors="coerce").fillna(1.0).clip(lower=1.0)
        # Bracketing tabulated durations; beyond the table the nearest pair extrapolates.
        grid = np.log(self.CDPP_DURATIONS_HR)
        lo = np.clip(np.searchsorted(grid, np.log(dur)) - 1, 0, len(grid) - 2)
        w = (np.log(dur) - grid[lo]) / (grid[lo + 1] - grid[lo])

        obs = pairs[pairs["n"] > 0]
        rows = obs["row"].to_numpy()
        sectors = obs["sector"].to_numpy()
        cdpp_lo, rank_lo = self._pair_cdpp(rows, sectors, lo, tmag, tic)
        cdpp_hi, rank_hi = self._pair_cdpp(rows, sectors, lo + 1, tmag, tic)
        cdpp = np.exp(np.log(cdpp_lo) + w[rows] * (np.log(cdpp_hi) - np.log(cdpp_lo)))
        rank = np.maximum(rank_lo, rank_hi)
        inv_var = np.bincount(rows, weights=obs["n"].to_numpy() / cdpp ** 2, minlength=n_rows)
        n_used = np.bincount(rows, weights=obs["n"].to_numpy(), minlength=n_rows)
        best = np.full(n_rows, len(self.NOISE_SOURCES), dtype=np.int64)
        np.minimum.at(best, rows, rank)

        snr = np.where(inv_var > 0, signal / dilution.to_numpy() * np.sqrt(inv_var), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.catalog["tess_cdpp_ppm"] = np.where(inv_var > 0, np.sqrt(n_used / inv_var), np.nan)
        self.catalog["tess_dilution"] = dilution
        self.catalog["tess_signal_rms_ppm"] = signal
        self.catalog["tess_cdpp_duration_hr"] = np.where(inv_var > 0, dur, np.nan)
        self.catalog["tess_noise_source"] = np.array(self.NOISE_SOURCES + ["no_transits"], dtype=object)[best]
        self.catalog["tess_snr"] = snr
        self.catalog["tess_snr_threshold"] = self.snr_threshold
        return self.catalog["tess_snr"]

    # ------------------------------------------------------------------
    # Optional magnitude feasibility gate
    # ------------------------------------------------------------------

    def _bright_enough(self) -> pd.Series:
        """Optional TESS magnitude cutoff; disabled by default."""
        bright = pd.Series(True, index=self.catalog.index)
        if self.tmag_limit is not None:
            bright = pd.to_numeric(self.catalog["tess_tmag"], errors="coerce").le(self.tmag_limit).fillna(False)
        self.catalog["tess_tmag_limit"] = self.tmag_limit
        self.catalog["tess_star_bright_enough"] = bright
        return bright

    def _classify_reasons(self) -> pd.Series:
        observed = self.catalog["tess_observed"].astype(bool)
        trans = self.catalog["tess_transiting_geometric"].astype(bool)
        bright = self.catalog["tess_star_bright_enough"].astype(bool)
        ntr = pd.to_numeric(self.catalog["tess_n_transits"], errors="coerce").fillna(0)
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
        """Run the toy TESS detector.

        Flat-universe detection uses four gates:
            1. observed by TESS
            2. transiting geometry / observed transit flag
            3. enough observed transits
            4. SNR >= threshold

        Calibration rows can use catalog sectors, measured depth/duration and SPOC CDPP,
        but the final logical detector is intentionally the same threshold model.
        """
        # 1. Observation window is already set during initialization.

        # 2. Transit geometry: P-Pop inclinations; NASA/TOI rows may use observed tran_flag.
        self.catalog["tess_transiting_geometric"] = self._transiting()
        self._impact_parameter()

        # 3. Count transits in the available sector windows.
        counts = self._transit_counts()

        # Optional magnitude feasibility; disabled by default.
        self.catalog["tess_star_bright_enough"] = self._bright_enough()

        # 4. Detection statistic: limb-darkened signal over per-sector CDPP.
        self.catalog["tess_transit_duration_hr"] = self._duration_hr()
        self.catalog["tess_transit_depth_ppm"] = self._depth_ppm()
        self.catalog["tess_snr"] = self._snr(counts)

        snr_col = pd.to_numeric(self.catalog["tess_snr"], errors="coerce")
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
        self._classify_reasons()
        return self.catalog

"""
07_rv_completeness_fix.py

PROPOSAL (do not edit the live model yet): fix the RV detector under-crediting the
brightest, nearest M-dwarf campaigns (Proxima b, Barnard b, GJ 367 b ... were
measured at 9-20 sigma but the model scored them 1.5-5 sigma and dropped them).

Diagnosis (see chat): NIRPS/HARPS band logic is already correct and NIRPS wins for
M dwarfs. The misses come from TWO survey assumptions that are too stingy for the
flagship targets, NOT from a physics error (the model's K matches real pl_rvamp 1.00):

  (1) N_obs is fixed at 100, but the flagship nearby M-dwarf programs took 200-500
      RV epochs. sigma_K = sqrt(2/N)*sigma_RV, so too-few epochs => sigma_K too big.
  (2) Stellar jitter is used RAW. Real teams GP-model the rotation-correlated part
      of the jitter and subtract it, so the EFFECTIVE jitter is much smaller.
  (2b, optional) The photon noise is floored at sigma_phot_ref even for very bright
      stars; Proxima (J=5.2) should beat 1 m/s, not floor at it.

This script implements all three as a thin RVData subclass (RVDataProposed) so the
LIVE model is untouched, then validates against the real measured planets split by
spectral type. The exact rv_data.py diff to apply later is at the bottom.

KEY COMPLETENESS POINT (the user's insight): N_obs in the model is ALREADY global
(FGK and M get the same number). Real telescope time is skewed toward M dwarfs, but
a *completeness* map should not inherit that bias -- give every star the same generous
N. The residual "M dwarfs are easier" that remains is real physics (K ~ M*^-2/3 P^-1/3),
not allocation.

Run:
    python scripts/07_rv_completeness_fix.py
    python scripts/07_rv_completeness_fix.py --n-obs 200 --gp 0.5
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("rv_data", ROOT / "detectors" / "rv_data.py")
_rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rv)
RVData = _rv.RVData

NASA_FILE = ROOT / "run" / "kepler" / "data" / "NASA" / "NASA_PSCompPars_rvamp_calibration.csv"


# --------------------------------------------------------------------------- #
# The proposed model, as a non-invasive subclass
# --------------------------------------------------------------------------- #

class RVDataProposed(RVData):
    """RVData + two survey-realism knobs (and one optional photon fix).

    jitter_gp_factor : multiply the per-type stellar jitter by this (GP activity
                       removal).  1.0 = current behaviour; ~0.5 = realistic GP cleaning.
    photon_beats_floor : if True, photon noise is allowed to drop BELOW sigma_phot_ref
                         for stars brighter than phot_full_mag (down to a hard systematic
                         floor sigma_phot_sys_floor), instead of clamping at the floor.
    n_obs is inherited unchanged -- pass a larger value (e.g. 200-300) at construction.
    """

    def __init__(self, *args, jitter_gp_factor: float = 1.0,
                 photon_beats_floor: bool = False,
                 sigma_phot_sys_floor: float = 0.15, **kwargs):
        super().__init__(*args, **kwargs)
        self.jitter_gp_factor = float(jitter_gp_factor)
        self.photon_beats_floor = bool(photon_beats_floor)
        self.sigma_phot_sys_floor = float(sigma_phot_sys_floor)
        # GP-discount the activity jitter (applied once, before any calc_noise call).
        self.jitter_by_stype = {k: v * self.jitter_gp_factor for k, v in self.jitter_by_stype.items()}

    def calc_noise(self) -> pd.Series:
        """Same as RVData.calc_noise but lets bright stars beat the photon floor."""
        mag = self.apparent_band_mag()
        if self.photon_beats_floor:
            excess = (mag - self.phot_full_mag)              # may be negative => brighter than floor
        else:
            excess = (mag - self.phot_full_mag).clip(lower=0.0)
        sigma_phot = (self.sigma_phot_ref_ms * 10 ** (0.2 * excess)).clip(lower=self.sigma_phot_sys_floor)
        sigma_phot = sigma_phot.replace([np.inf, -np.inf], np.nan).fillna(self.sigma_phot_ref_ms)
        jitter = self.catalog["stype"].map(self.jitter_by_stype).astype(float)
        jitter = jitter.fillna(self.jitter_by_stype.get("Unknown", 2.0))
        sigma_rv = np.sqrt(self.sigma_instr_ms ** 2 + sigma_phot ** 2 + jitter ** 2)
        self.catalog["rv_sigma_phot_ms"] = sigma_phot
        self.catalog["rv_sigma_jitter_ms"] = jitter
        self.catalog["rv_sigma_ms"] = sigma_rv
        return sigma_rv


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_real_planets() -> pd.DataFrame:
    df = pd.read_csv(NASA_FILE)
    for c in ["pl_rade", "pl_bmasse", "pl_orbper", "st_mass", "st_teff", "st_lum",
              "sy_dist", "pl_rvamp", "pl_rvamperr1", "pl_rvamperr2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["l_sun"] = 10 ** df["st_lum"]
    m = (df.pl_bmassprov.isin(["Mass", "Msini", "Msin(i)/sin(i)"])
         & (df.pl_rade <= 2.2) & (df.pl_bmasse <= 12) & (df.pl_rade > 0)
         & df.pl_orbper.notna() & df.st_mass.notna() & df.pl_rvamp.notna())
    sub = df[m].rename(columns={
        "pl_rade": "radius_p", "pl_bmasse": "mass_p", "pl_orbper": "p_orb",
        "st_mass": "mass_s", "st_teff": "teff_s", "sy_dist": "distance_s"}).copy()
    sub["inc_p"] = 90.0
    sub["ecc_p"] = 0.0
    sub["stype"] = pd.cut(sub.teff_s, [0, 3700, 5200, 6000, 1e4], labels=["M", "K", "G", "F"])
    kerr = pd.concat([sub.pl_rvamperr1.abs(), sub.pl_rvamperr2.abs()], axis=1).mean(axis=1)
    sub["real_sig"] = sub.pl_rvamp / kerr     # real published K significance
    return sub


# --------------------------------------------------------------------------- #
# Recovery (best of HARPS/NIRPS)
# --------------------------------------------------------------------------- #

def recovered(planets, cls, n_obs, mag_target, snr_thr, **extra):
    """best(HARPS,NIRPS): detected if either band reaches snr>=thr AND mag<=mag_target."""
    ok = np.zeros(len(planets), bool)
    for inst in ("HARPS", "NIRPS"):
        c = cls(planets.copy(), source="pscomppars", instrument=inst, n_obs=n_obs,
                snr_threshold=snr_thr, apply_sini=True, validate_for_detection=False,
                **extra).determine_detectable()
        mag = pd.to_numeric(c["rv_mag"], errors="coerce").to_numpy()
        snr = pd.to_numeric(c["rv_snr"], errors="coerce").to_numpy()
        ok |= (mag <= mag_target) & (snr >= snr_thr)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # DEFAULT = the safe fix: photon_beats_floor only (steepen the brightness slope),
    # keep N and jitter as-is. A global N/jitter boost OVER-corrects (see chat), so the
    # photon change is the core recommendation; --n-obs / --gp are optional knobs.
    ap.add_argument("--n-obs", type=int, default=100, help="global N_obs (raise only if you want yield, not completeness)")
    ap.add_argument("--gp", type=float, default=1.0, help="jitter_gp_factor (1.0 = unchanged; ~0.5 = GP removal)")
    ap.add_argument("--mag-target", type=float, default=14.0)
    ap.add_argument("--snr-threshold", type=float, default=5.0)
    args = ap.parse_args()

    if not NASA_FILE.exists():
        print(f"ERROR: need {NASA_FILE} (the pl_rvamp calibration pull).")
        sys.exit(1)
    real = load_real_planets()
    print(f"Real small (R<=2.2, M<=12) measured-mass planets with pl_rvamp: {len(real)}")
    print(f"PROPOSED: N_obs={args.n_obs} (global, same for FGK & M), jitter_gp_factor={args.gp}, "
          f"photon_beats_floor=True\n")

    base_ok = recovered(real, RVData, 100, args.mag_target, args.snr_threshold)
    prop_ok = recovered(real, RVDataProposed, args.n_obs, args.mag_target, args.snr_threshold,
                        jitter_gp_factor=args.gp, photon_beats_floor=True)
    real_ok = (real.real_sig >= args.snr_threshold).to_numpy()

    print(f"{'stype':<7}{'N':>5}{'BASELINE':>11}{'PROPOSED':>11}{'REAL(target)':>14}")
    print("-" * 48)
    for st in ["F", "G", "K", "M", "ALL"]:
        sel = np.ones(len(real), bool) if st == "ALL" else (real.stype == st).to_numpy()
        n = int(sel.sum())
        if n == 0:
            continue
        print(f"{st:<7}{n:>5}{base_ok[sel].mean():>10.0%}{prop_ok[sel].mean():>11.0%}"
              f"{real_ok[sel].mean():>13.0%}")
    print(f"\n(REAL(target) = fraction the archive PUBLISHED at >= {args.snr_threshold:g} sigma. This is a"
          "\n YIELD lower bound -- not every star gets a campaign -- so a COMPLETENESS map SHOULD sit at or"
          "\n above it. The real test is per-planet ranking: genuinely-easy bright planets (Proxima, real"
          "\n ~20 sigma) must be recovered, while low-real-sigma planets must NOT be over-recovered.)")

    # Per-planet RANKING test: split by REAL significance. A good fix recovers the
    # genuinely-easy (real_sig>=5) planets without over-recovering the genuinely-hard ones.
    easy = real_ok                                  # real_sig >= threshold
    hard = ~real_ok
    print("\nPer-planet ranking (the real acid test):")
    print(f"  genuinely-easy (real>= {args.snr_threshold:g}sig, N={easy.sum()}): "
          f"baseline recovers {base_ok[easy].mean():.0%} -> proposed {prop_ok[easy].mean():.0%}  (want ~100%)")
    print(f"  genuinely-hard (real<  {args.snr_threshold:g}sig, N={hard.sum()}): "
          f"baseline 'recovers' {base_ok[hard].mean():.0%} -> proposed {prop_ok[hard].mean():.0%}  "
          f"(want LOW = few false positives)")

    # Per-planet: the bright M dwarfs that the baseline dropped.
    mdw = real[real.stype == "M"].copy()
    mdw["baseline"] = base_ok[(real.stype == "M").to_numpy()]
    mdw["proposed"] = prop_ok[(real.stype == "M").to_numpy()]
    fixed = mdw[(~mdw.baseline) & mdw.proposed].sort_values("real_sig", ascending=False)
    print(f"\nBright M-dwarf planets RECOVERED by the proposal that the baseline missed "
          f"({len(fixed)}):")
    show = [c for c in ["pl_name", "flux_p", "mass_p", "p_orb", "pl_rvamp", "real_sig"] if c in fixed.columns]
    if len(fixed):
        print(fixed[show].head(20).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    still = mdw[~mdw.proposed]
    print(f"\nStill missed M dwarfs ({len(still)}): "
          f"{', '.join(still.get('pl_name', pd.Series([], dtype=str)).head(10))}")

    print("\n" + EXACT_DIFF)


EXACT_DIFF = r"""
================================ EXACT rv_data.py DIFF (apply when free) ================================
CORE CHANGE (recommended): let bright stars beat the photon floor. This fixes the model's
too-FLAT brightness slope -- it under-credited Proxima/GJ 1002 (real 9-20 sigma, modeled <5).
It lifts genuinely-easy planets (real>=5 sigma) 79% -> 87% while adding almost no false positives
on the genuinely-hard ones (57% -> 59%). The K (signal) side is untouched (already matches pl_rvamp).

1) __init__ signature -- add two params (backward-compatible):
       photon_beats_floor: bool = False,     # let bright stars beat the photon floor; SET True
       sigma_phot_sys_floor: float = 0.15,   # irreducible photon/systematic floor (m/s)

2) End of __init__:
       self.photon_beats_floor = bool(photon_beats_floor)
       self.sigma_phot_sys_floor = float(sigma_phot_sys_floor)

3) calc_noise() -- replace the photon-excess clamp:
       -        excess = (mag - self.phot_full_mag).clip(lower=0.0)
       -        sigma_phot = self.sigma_phot_ref_ms * 10 ** (0.2 * excess)
       +        excess = (mag - self.phot_full_mag)
       +        if not getattr(self, "photon_beats_floor", False):
       +            excess = excess.clip(lower=0.0)
       +        sigma_phot = (self.sigma_phot_ref_ms * 10 ** (0.2 * excess)
       +                      ).clip(lower=getattr(self, "sigma_phot_sys_floor", 0.15))

OPTIONAL knobs (do NOT enable by default -- they OVER-correct):
   - jitter_gp_factor (multiply jitter, ~0.5 for GP cleaning) and a larger global n_obs raise
     recovery EVERYWHERE, pushing ALL recovery to ~92% vs the real ~59% published yield. Only use
     them if you specifically want to model an intensive (200-500 epoch) campaign, not standard
     completeness.

WHY "recovery > real yield" is OK: REAL(target) is YIELD (not every star is observed); a
COMPLETENESS map should sit at or ABOVE it. Validate by per-planet RANKING (recover real>=5 sigma,
do not over-recover real<5 sigma), NOT by matching the yield fraction.

KNOWN RESIDUAL (separate, opposite issue): the model is slightly too GENEROUS for hard FGK small
planets (F: model 33% vs real-yield 11%). That is the faint/hard end, not fixed here; it would need
a steeper faint-end noise rise, and is the opposite of the bright-M-dwarf problem.
========================================================================================================
"""


if __name__ == "__main__":
    main()

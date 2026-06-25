"""
apply_detection_calibration.py — fold the official-pipeline MES/SNR calibration
into the already-generated Kepler & TESS catalogs, in place.

Background
----------
scripts/47 and scripts/48 measured that the toy boxcar detectors ran ~1.2x hot
versus the official Kepler DR25 MES and TESS SPOC SNR, so the toy crossed the
7.1 detection threshold too readily and over-detected marginal planets.  The fix
adds a multiplicative calibration factor to the detector formulas:

    KeplerData.MES_OFFICIAL_CALIBRATION   (~1/1.19)        -> kepler_mes
    TESSData.SNR_OFFICIAL_CALIBRATION     (~1/1.52 @ thr)  -> tess_snr

If a catalog was already calibrated with a different factor, the `detection_calibration`
stamp lets this script apply only the *delta* (target / prior), so re-running after a
factor change re-calibrates correctly instead of double-applying.

The calibration is a *pure scalar* on the detection statistic — nothing else in
the pipeline (transit depth, CDPP, transit count, geometry, sector phase)
changes.  So rather than regenerate the catalogs (which for TESS would re-roll
the random transit phase and change results beyond the calibration), we simply:

    1. multiply the stored kepler_mes / tess_snr by the calibration factor,
    2. rescale derived noise columns that scale with it,
    3. recompute the boolean detected* columns from the unchanged gate columns
       AND the rescaled statistic >= 7.1.

This is exact (the threshold model was verified to reproduce detected* to 100%)
and idempotent (a `detection_calibration` stamp column guards against
double-application).

Run from repo root:
    python run/apply_detection_calibration.py
    python run/apply_detection_calibration.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_attr(modfile: Path, clsname: str, attr: str) -> float:
    """Load a class constant without importing the lifesim package (PyQt-free)."""
    spec = importlib.util.spec_from_file_location(modfile.stem + "_standalone", modfile)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return float(getattr(getattr(mod, clsname), attr))


CAL_K = _load_attr(ROOT / "lifesim" / "core" / "kepler_data.py", "KeplerData", "MES_OFFICIAL_CALIBRATION")
CAL_T = _load_attr(ROOT / "lifesim" / "core" / "tess_data.py", "TESSData", "SNR_OFFICIAL_CALIBRATION")

# (label, directory, glob) for every catalog set that stores a detection statistic.
KEPLER_DIRS = [
    ("kepler/Gaia_C_F_K_combined", ROOT / "run" / "kepler" / "data" / "Gaia_C_F_K_combined", "kepler_catalog_*.csv"),
    ("kepler/Gaia",                ROOT / "run" / "kepler" / "data" / "Gaia",                "kepler_catalog_*.csv"),
]
TESS_DIRS = [
    ("tess/Gaia_C_F_K_combined_cdpp_v1", ROOT / "run" / "tess" / "data" / "Gaia_C_F_K_combined_cdpp_v1", "tess_catalog_*.csv"),
    ("tess/Gaia_cdpp_v1",                ROOT / "run" / "tess" / "data" / "Gaia_cdpp_v1",                "tess_catalog_*.csv"),
]

STAMP_COL = "detection_calibration"


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes", "t"])


def _and_gates(df: pd.DataFrame, cols: list[str], statistic: pd.Series, threshold: pd.Series) -> pd.Series:
    out = statistic >= threshold
    for c in cols:
        if c in df.columns:
            out = out & _bool(df[c])
    return out.fillna(False)


def rescale_kepler(df: pd.DataFrame, factor: float, target: float) -> tuple[pd.DataFrame, bool]:
    if "kepler_mes" not in df.columns:
        return df, False
    df = df.copy()
    df["kepler_mes"] = pd.to_numeric(df["kepler_mes"], errors="coerce") * factor
    if "min_detectable_depth_ppm" in df.columns:   # = thr * cdpp/(sqrtN*MES_CORR*cal) -> /cal
        df["min_detectable_depth_ppm"] = pd.to_numeric(df["min_detectable_depth_ppm"], errors="coerce") / factor

    thr = pd.to_numeric(df.get("kepler_mes_threshold", pd.Series(7.1, index=df.index)), errors="coerce").fillna(7.1)
    base = ["transiting_geometric", "kepler_enough_transits"]
    variants = {
        "detected":       "kepler_star_bright_enough",
        "detected_best":  "kepler_star_bright_enough_best",
        "detected_worst": "kepler_star_bright_enough_worst",
    }
    for col, bright in variants.items():
        if col in df.columns:
            df[col] = _and_gates(df, base + [bright], df["kepler_mes"], thr)
    if "kepler_p_detect" in df.columns and "detected" in df.columns:
        df["kepler_p_detect"] = df["detected"].astype(float)
    df[STAMP_COL] = target
    return df, True


def rescale_tess(df: pd.DataFrame, factor: float, target: float) -> tuple[pd.DataFrame, bool]:
    if "tess_snr" not in df.columns:
        return df, False
    df = df.copy()
    df["tess_snr"] = pd.to_numeric(df["tess_snr"], errors="coerce") * factor

    thr = pd.to_numeric(df.get("tess_snr_threshold", pd.Series(7.1, index=df.index)), errors="coerce").fillna(7.1)
    gates = ["tess_observed", "tess_transiting_geometric", "tess_star_bright_enough", "tess_enough_transits"]
    detected = _and_gates(df, gates, df["tess_snr"], thr)
    for col in ["detected", "tess_detected", "detected_best", "detected_worst"]:
        if col in df.columns:
            df[col] = detected
    if "tess_p_detect" in df.columns:
        df["tess_p_detect"] = detected.astype(float)
    df[STAMP_COL] = target
    return df, True


def _prior_stamp(path: Path) -> float:
    """Calibration factor already baked into this file (1.0 if never calibrated)."""
    head = pd.read_csv(path, nrows=1)
    if STAMP_COL not in head.columns:
        return 1.0
    val = pd.to_numeric(head[STAMP_COL], errors="coerce").iloc[0]
    return float(val) if np.isfinite(val) else 1.0


def _det_frac(df: pd.DataFrame, det_col: str) -> float:
    if det_col not in df.columns:
        return float("nan")
    trans = _bool(df["transiting_geometric"]) if "transiting_geometric" in df.columns else pd.Series(True, index=df.index)
    m = trans
    return (_bool(df[det_col])[m].mean() * 100.0) if m.any() else float("nan")


def process(dirs, cal, rescale_fn, det_col, dry_run: bool) -> None:
    for label, d, pattern in dirs:
        files = sorted(d.glob(pattern))
        if not files:
            print(f"  [{label}] no files in {d}")
            continue
        n_done = n_skip = 0
        before_acc, after_acc = [], []
        for f in files:
            prior = _prior_stamp(f)
            if abs(prior - cal) < 1e-6:           # already at the target calibration
                n_skip += 1
                continue
            factor = cal / prior                  # delta to apply now (re-calibration-safe)
            df = pd.read_csv(f)
            before_acc.append(_det_frac(df, det_col))
            out, changed = rescale_fn(df, factor, cal)
            if not changed:
                n_skip += 1
                continue
            after_acc.append(_det_frac(out, det_col))
            if not dry_run:
                fd, tmp = tempfile.mkstemp(suffix=".csv", dir=str(d))
                os.close(fd)
                out.to_csv(tmp, index=False)
                os.replace(tmp, f)
            n_done += 1
        b = np.nanmean(before_acc) if before_acc else float("nan")
        a = np.nanmean(after_acc) if after_acc else float("nan")
        tag = "DRY-RUN " if dry_run else ""
        print(f"  [{label}] {tag}cal={cal:.3f}  files={len(files)}  rescaled={n_done}  skipped(done)={n_skip}  "
              f"detected%(transiting): {b:.1f}% -> {a:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report effect without writing files")
    ap.add_argument("--mission", choices=["kepler", "tess", "both"], default="both")
    args = ap.parse_args()

    print(f"Kepler MES calibration: x{CAL_K:.3f}   TESS SNR calibration: x{CAL_T:.3f}")
    print(f"{'(dry run — no files written)' if args.dry_run else 'Applying in place (atomic writes)'}\n")

    if args.mission in ("kepler", "both"):
        print("Kepler:")
        process(KEPLER_DIRS, CAL_K, rescale_kepler, "detected", args.dry_run)
    if args.mission in ("tess", "both"):
        print("TESS:")
        process(TESS_DIRS, CAL_T, rescale_tess, "detected", args.dry_run)

    print("\nDone." + ("" if args.dry_run else "  Re-run scripts/44 to see the calibrated detection background."))


if __name__ == "__main__":
    main()

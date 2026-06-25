"""
recompute_kepler_detection.py — re-apply the Kepler detection model in place.

Only the DETECTION noise model changed (kepler_data.py: a stellar-variability
CDPP floor added in quadrature, cdpp_variability_ppm=28). The P-Pop planet + star
parameters in the stored catalogs are unchanged, so there is no need to
regenerate anything. We simply re-run KeplerData.determine_detectable() on each
stored catalog, which recomputes kepler_cdpp_ppm / kepler_mes / detected with the
fixed model, and write the result back.

Why this is safe:
  * Deterministic & idempotent — detection is recomputed from the preserved
    radius_p / radius_s / kepler_mag_used / p_orb / inc_p / ... columns, so
    running it once or ten times gives the same answer.
  * Same code path as generation — KeplerData(df) infers source='ppop' for these
    catalogs (no NASA columns), exactly as run_single did, so the output matches
    what a full regeneration would produce for the detection columns.
  * Atomic writes — each catalog is written to a temp file then os.replace()'d,
    so an interruption cannot leave a half-written CSV.

Run from repo root:
    python run/kepler/recompute_kepler_detection.py
"""

import os
import sys
import importlib.util
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# Load KeplerData directly from its file instead of `from lifesim.core.kepler_data
# import KeplerData`. The lifesim package __init__ eagerly imports the GUI
# Instrument (PyQt5/spectres), which is irrelevant to this headless recompute and
# may be absent. kepler_data.py itself only needs numpy/pandas (its lifesim.core
# imports are wrapped in try/except), so this loads cleanly on its own.
def _load_kepler_data_class():
    path = ROOT / "lifesim" / "core" / "kepler_data.py"
    spec = importlib.util.spec_from_file_location("kepler_data_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.KeplerData

KeplerData = _load_kepler_data_class()

KEPLER_DIR = ROOT / "run" / "kepler" / "data" / "Gaia"
N_UNIVERSES = 10


def _as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def _detected_pct_by_type(df: pd.DataFrame) -> dict:
    """Detected fraction among transiting planets, per spectral type."""
    st = df["stype"].astype(str).str[0]
    trans = _as_bool(df["transiting_geometric"])
    det = _as_bool(df["detected"])
    out = {}
    for t in ["F", "G", "K", "M"]:
        m = (st == t) & trans
        out[t] = (det[m].mean() * 100.0) if m.any() else float("nan")
    return out


def main():
    files = [KEPLER_DIR / f"kepler_catalog_{i}.csv" for i in range(N_UNIVERSES)]
    files = [f for f in files if f.exists()]
    if not files:
        print(f"No kepler_catalog_*.csv found in {KEPLER_DIR}")
        return

    print(f"Re-running Kepler detection on {len(files)} catalog(s) "
          f"(cdpp_variability_ppm=28, quadrature floor)\n")
    print(f"{'file':24s} {'rows':>9s}   detected% among transiting (F / G / K / M)")
    print("-" * 78)

    for f in files:
        df = pd.read_csv(f)
        before = _detected_pct_by_type(df)

        kd = KeplerData(df)            # source='auto' -> 'ppop' (same as generation)
        kd.determine_detectable()      # recompute CDPP / MES / detected with the fix
        out = kd.catalog
        after = _detected_pct_by_type(out)

        # Atomic in-place write (temp file on same filesystem, then replace).
        fd, tmp = tempfile.mkstemp(suffix=".csv", dir=str(KEPLER_DIR))
        os.close(fd)
        out.to_csv(tmp, index=False)
        os.replace(tmp, f)

        deltas = "  ".join(
            f"{t}:{before[t]:4.0f}%->{after[t]:4.0f}%" for t in ["F", "G", "K", "M"]
        )
        print(f"{f.name:24s} {len(out):9,d}   {deltas}")

    print("\nDone. All catalogs updated in place. Re-run script 44 to see the corrected FGK background.")


if __name__ == "__main__":
    main()

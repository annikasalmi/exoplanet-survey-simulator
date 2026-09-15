"""Re-run KeplerData.determine_detectable() on the stored Kepler catalogs after a detection-model
change, without regenerating planets. Deterministic, and writes each catalog atomically.
Run from repo root: python run/kepler/recompute_kepler_detection.py
"""

import os
import sys
from tools.paths import KEPLER_DATA_DIR
import importlib.util
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# This file-path loader dates from when kepler_data.py lived in lifesim/core/, so
# importing it dragged in the lifesim package __init__ and with it the GUI
# Instrument (PyQt5/spectres). It now lives in telescopes/kepler/, whose __init__ imports
# nothing, so `from telescopes.kepler.detection_model import KeplerData` would do just as well.
def _load_kepler_data_class():
    path = ROOT / "telescopes" / "kepler" / "detection_model.py"
    spec = importlib.util.spec_from_file_location("kepler_data_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.KeplerData

KeplerData = _load_kepler_data_class()

KEPLER_DIR = Path(KEPLER_DATA_DIR) / "Gaia"
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
          f"(CDPP fallback {KeplerData.CDPP_NONSTELLAR_KP12_PPM} ppm @Kp12 (+) "
          f"{KeplerData.CDPP_STELLAR_PPM} ppm stellar floor)\n")
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

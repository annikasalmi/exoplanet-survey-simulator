"""Re-run KeplerData.determine_detectable() on the stored Kepler catalogs after a detection-model
change, without regenerating planets. Deterministic, and writes each catalog atomically.
Run from repo root: python run/kepler/recompute_kepler_detection.py
"""

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from tools.paths import KEPLER_DATA_DIR
from science.telescopes.detection import as_boolean
from science.telescopes.kepler.detection_model import KeplerData

KEPLER_DIR = Path(KEPLER_DATA_DIR) / "Gaia"
N_UNIVERSES = 10


def _detected_pct_by_type(df: pd.DataFrame) -> dict:
    """Detected fraction among transiting planets, per spectral type."""
    st = df["stype"].astype(str).str[0]
    trans = as_boolean(df["transiting_geometric"])
    det = as_boolean(df["detected"])
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

    print("\nDone. All catalogs updated in place. Re-run plotting/scripts/analysis/rocky_scatter_gaia60pc.py to see the corrected FGK background.")


if __name__ == "__main__":
    main()

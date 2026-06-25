"""
# =============================================================================
# Build a volume-limited Gaia DR3 star catalog within 60 pc.
# =============================================================================

Run ONCE (needs internet) to (re)create `gaia_within_60pc.csv`:

    python PPop/StarCatalogs/build_gaia_60pc.py

Why a custom export instead of the old 20 pc file:
- The old `gaia_within_20pc.csv` + loader required teff_gspphot, radius_gspphot
  AND mass_flame for every star.  Those Gaia astrophysical products are missing
  for most faint M dwarfs, so the old sample dropped ~83% of nearby stars and was
  badly M-dwarf-incomplete.
- Here we pull only the always-present astrometry/photometry (parallax, G, BP-RP)
  plus the GSP-Phot / FLAME parameters WHEN available.  The loader (gaia.py) then
  fills any missing Teff / radius / mass from BP-RP main-sequence relations, so no
  star is dropped for lacking a derived parameter.

Distance limit: parallax > 16.667 mas  <=>  d < 60 pc.
"""

import os
import time
import numpy as np
import pandas as pd

from astroquery.gaia import Gaia

# The archive caps anonymous sync jobs at 2000 rows; -1 lifts our client-side cap so
# the async job returns the full ~59k-row result set.
Gaia.ROW_LIMIT = -1

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gaia_within_60pc.csv")

# parallax > 1000/60 mas = 60 pc.  parallax_over_error > 5 keeps distances trustworthy
# without throwing away faint M dwarfs.  ruwe < 1.4 keeps well-behaved single-star
# astrometry.  We require a BP-RP colour so the loader can always derive Teff/R/M.
# We deliberately pull only columns that always exist in gaiadr3.gaia_source.
# Teff / radius / mass are NOT requested: the GSP-Phot / FLAME astrophysical
# products live in a separate table and are missing for most faint M dwarfs, so
# relying on them is exactly what made the old 20 pc sample M-dwarf-incomplete.
# gaia.py derives Teff/radius/mass from BP-RP + absolute G instead, for every star.
ADQL = """
SELECT source_id, ra, dec, parallax, parallax_error,
       phot_g_mean_mag, bp_rp, ruwe
FROM gaiadr3.gaia_source
WHERE parallax > 16.667
  AND parallax_over_error > 5
  AND ruwe < 1.4
  AND phot_g_mean_mag IS NOT NULL
  AND bp_rp IS NOT NULL
"""


def main():
    print("--> Querying Gaia DR3 archive for sources within 60 pc ...")
    # The pre-DR4 archive is flaky and intermittently returns HTTP 500; retry.
    last_err = None
    for attempt in range(1, 6):
        try:
            job = Gaia.launch_job_async(ADQL)
            tbl = job.get_results()
            break
        except Exception as e:  # noqa: BLE001 - archive throws assorted transient errors
            last_err = e
            print(f"    attempt {attempt}/5 failed ({type(e).__name__}); retrying in 15 s ...")
            time.sleep(15)
    else:
        raise RuntimeError(f"Gaia archive query failed after 5 attempts: {last_err}")
    df = tbl.to_pandas()
    print(f"--> Retrieved {len(df)} sources")

    # Normalise column names to what gaia.py expects.
    df = df.rename(columns={"source_id": "SOURCE_ID"})

    keep = ["SOURCE_ID", "ra", "dec", "parallax", "parallax_error",
            "phot_g_mean_mag", "bp_rp", "ruwe"]
    df = df[keep]

    df.to_csv(OUT, index=False)
    print(f"--> Wrote {OUT}  ({len(df)} stars)")

    # Quick report.
    dist = 1000.0 / df["parallax"].to_numpy()
    print(f"--> Distance range: {dist.min():.1f} - {dist.max():.1f} pc")
    print(f"--> BP-RP range:    {df['bp_rp'].min():.2f} - {df['bp_rp'].max():.2f}")


if __name__ == "__main__":
    main()

"""Build the two small reference files TESSData reads by default, both under telescopes/tess/data/:

  sector_grid.npz       which TESS sectors (1..MAX_SECTOR) cover each cell of an equal-area sky
                        grid, from tess-point. Looking stars up in it replaces a per-star tess-point
                        call, which costs ~6 ms per star.
  spoc_cdpp_tmag.csv    median SPOC robust RMS CDPP per sector and 0.5-mag Tmag bin, for every
                        2-min target in the SPOC CDPP tables (Twicken et al. 2025, RNAAS 9, 132),
                        plus all-sector rows (sector 0). Built from the per-sector MAST files in
                        results/catalogs/tess/CDPP (MAST TCE bulk-download page).

Run from repo root:  python telescopes/tess/build_reference_data.py [grid|noise|all] [cdpp_dir]
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paths import TESS_DATA_DIR

OUT_DIR = Path(__file__).resolve().parent / "data"
MAX_SECTOR = 106            # last sector with a SPOC CDPP table on MAST when this was built
GRID_N_LON = 360            # 1 deg in ecliptic longitude
GRID_N_SINLAT = 180         # equal steps in sin(ecliptic latitude), so every cell has equal area
OBLIQUITY_DEG = 23.439
TMAG_BIN = 0.5
MIN_STARS_PER_BIN = 10


def grid_cell_centers():
    """Ecliptic lon/lat (deg) of every grid cell centre, cell index = i_sinlat * GRID_N_LON + i_lon."""
    lon = (np.arange(GRID_N_LON) + 0.5) * 360.0 / GRID_N_LON
    sinlat = -1.0 + (np.arange(GRID_N_SINLAT) + 0.5) * 2.0 / GRID_N_SINLAT
    lon2, sinlat2 = np.meshgrid(lon, sinlat)
    return lon2.ravel(), np.rad2deg(np.arcsin(sinlat2.ravel()))


def ecliptic_to_equatorial(lon_deg, lat_deg):
    lam, beta, eps = np.deg2rad(lon_deg), np.deg2rad(lat_deg), np.deg2rad(OBLIQUITY_DEG)
    dec = np.arcsin(np.sin(beta) * np.cos(eps) + np.cos(beta) * np.sin(eps) * np.sin(lam))
    ra = np.arctan2(np.sin(lam) * np.cos(eps) - np.tan(beta) * np.sin(eps), np.cos(lam))
    return np.mod(np.rad2deg(ra), 360.0), np.rad2deg(dec)


def build_grid(chunk=5000):
    import tess_stars2px
    from importlib.metadata import version

    lon, lat = grid_cell_centers()
    ra, dec = ecliptic_to_equatorial(lon, lat)
    covered = np.zeros((len(ra), MAX_SECTOR), dtype=bool)
    t0 = time.time()
    for start in range(0, len(ra), chunk):
        ids = np.arange(start, min(start + chunk, len(ra)))
        out = tess_stars2px.tess_stars2px_function_entry(ids, ra[ids], dec[ids])
        sid, sec = np.asarray(out[0]), np.asarray(out[3])
        keep = (sec >= 1) & (sec <= MAX_SECTOR)
        covered[sid[keep], sec[keep] - 1] = True
        print(f"  grid cells {ids[-1] + 1:,}/{len(ra):,}  ({time.time() - t0:.0f}s)", flush=True)

    OUT_DIR.mkdir(exist_ok=True)
    np.savez_compressed(
        OUT_DIR / "sector_grid.npz",
        covered_bits=np.packbits(covered, axis=1),
        n_lon=GRID_N_LON, n_sinlat=GRID_N_SINLAT, max_sector=MAX_SECTOR,
        obliquity_deg=OBLIQUITY_DEG, tess_point_version=version("tess-point"),
    )
    n = covered.sum(axis=1)
    print(f"Saved sector_grid.npz: {np.mean(n > 0):.1%} of sky covered through S{MAX_SECTOR}, "
          f"median {np.median(n[n > 0]):.0f} sectors where covered")


def build_noise(cdpp_dir=Path(TESS_DATA_DIR) / "CDPP"):
    from telescopes.tess.detection_model import TESSData

    tab = TESSData._load_cdpp_tables(TESSData.__new__(TESSData), Path(cdpp_dir))
    if tab.empty:
        raise FileNotFoundError(f"No SPOC CDPP CSVs in {cdpp_dir}")
    tab = tab[tab["sector"] <= MAX_SECTOR].dropna(subset=["tmag"])
    cols = list(TESSData.CDPP_COLS.values())
    tab["tmag_bin"] = np.round(tab["tmag"] / TMAG_BIN) * TMAG_BIN

    per_sector = tab.groupby(["sector", "tmag_bin"])[cols].median()
    per_sector["n_stars"] = tab.groupby(["sector", "tmag_bin"]).size()
    all_sectors = tab.groupby("tmag_bin")[cols].median()
    all_sectors["n_stars"] = tab.groupby("tmag_bin").size()
    all_sectors = pd.concat({0: all_sectors}, names=["sector"])

    out = pd.concat([all_sectors, per_sector]).reset_index()
    out = out[out["n_stars"] >= MIN_STARS_PER_BIN]
    out["sector"] = out["sector"].astype(int)
    OUT_DIR.mkdir(exist_ok=True)
    out.round(3).to_csv(OUT_DIR / "spoc_cdpp_tmag.csv", index=False)
    print(f"Saved spoc_cdpp_tmag.csv: {len(out):,} rows from {len(tab):,} target-sectors, "
          f"sectors {tab['sector'].min()}-{tab['sector'].max()}, "
          f"Tmag {out['tmag_bin'].min()}-{out['tmag_bin'].max()}")


def print_tmag_corrections():
    """Median mbol - Tmag and G - Tmag per Teff bin over dwarf planet hosts in NASA PSCompPars,
    for TESSData._MBOL_MINUS_T and _G_MINUS_T (sy_tmag there is TIC-8, Stassun et al. 2019).
    """
    from telescopes.tess.detection_model import TESSData
    from tools.exoplanet_catalog import read_nasa_csv
    from tools.paths import PSCOMPPARS_CSV

    cols = ["st_teff", "st_lum", "sy_dist", "sy_tmag", "sy_gaiamag", "st_logg"]
    d = read_nasa_csv(PSCOMPPARS_CSV).drop_duplicates("hostname")[cols].apply(pd.to_numeric, errors="coerce")
    d = d.dropna(subset=["st_teff", "st_lum", "sy_dist", "sy_tmag"])
    d = d[(d["st_logg"] >= 4.0) | d["st_logg"].isna()]
    mbol = 4.74 - 2.5 * d["st_lum"] + 5 * np.log10(d["sy_dist"] / 10)
    grid = TESSData._TEFF_GRID
    edges = np.r_[2250, (grid[1:] + grid[:-1]) / 2, 9000]
    b = pd.cut(d["st_teff"], edges, labels=grid)
    out = pd.DataFrame({"n": d.groupby(b, observed=False).size(),
                        "mbol_minus_t": (mbol - d["sy_tmag"]).groupby(b, observed=False).median(),
                        "g_minus_t": (d["sy_gaiamag"] - d["sy_tmag"]).groupby(b, observed=False).median()})
    print(f"{len(d):,} dwarf hosts\n{out.round(2).to_string()}")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what == "tmag":
        print_tmag_corrections()
    if what in {"noise", "all"}:
        build_noise(*sys.argv[2:3])
    if what in {"grid", "all"}:
        build_grid()

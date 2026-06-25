#!/usr/bin/env python
"""
generate_gk_8000.py — top up the Gaia-60pc P-Pop background for G and K stars
until BOTH Kepler and TESS have >= 8000 transiting-in-window planets per type.

Why: script 44's per-type detection background only smooths once a panel holds
>= ~3000 transiting planets. Current Gaia-60pc stacks (universes 0..9) hold only
~1128 G / ~3026 K (Kepler) and ~1124 G / ~3009 K (TESS) — G is far short. This
script generates ADDITIONAL seeded universes (same config as run_sim.py:
Gaia-60pc star catalog + Bergsten2022 FGK occurrence) and runs the Kepler and
TESS detectors on them, writing new catalogs WITHOUT touching the existing ones.

Key choices (efficiency + correctness):
  * "transiting-in-window" matches script 44 exactly:
        Kepler : transiting_geometric
        TESS   : tess_observed & tess_transiting_geometric
        window : flux_p in (0.1, 1e4),  radius_p in (0.6, 2.2),  stype == G/K
  * Kepler and TESS universe i share ONE P-Pop draw (seed i): we generate the
    P-Pop catalogue once and run BOTH detectors on a copy -> ~half the cost of
    re-running run_Kepler and run_TESS separately.
  * We generate G-ONLY and K-ONLY universes (the Gaia catalogue is filtered to
    one spectral type per universe) so we don't waste compute on the ~45 extra
    K-universes that a "full universe until G is done" loop would produce.
  * New files are numbered from 8001 upward (kepler_catalog_8001.csv,
    tess_catalog_8001.csv, ...), so existing 0..9 are never overwritten.
  * Restartable: on startup it recounts every existing catalog (0..9 and 8001+)
    and only generates what is still missing. Killing and re-running resumes.

NOTE: to make script 44 (and friends) actually STACK these new files, raise its
N_UNIVERSES / file-discovery to include indices 8001+ (it currently reads 0..9).
This script only produces the catalogues.

Run from repo root:
    python run/generate_gk_8000.py                 # full job (long: ~60 G + ~17 K universes)
    python run/generate_gk_8000.py --types K       # only top up K
    python run/generate_gk_8000.py --smoke         # fast self-test (temp dirs, tiny)
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import warnings
warnings.filterwarnings("ignore")

import PPop.SystemGenerator as SystemGenerator
from PPop.StarCatalogs import gaia
from run.ppop.ppop_generator import PPop
from lifesim.core.kepler_data import KeplerData
from lifesim.core.tess_data import TESSData
from run.tess.run_TESS import _build_tess_kwargs

# ── Config ────────────────────────────────────────────────────────────────────

KEP_DIR = ROOT / "run" / "kepler" / "data" / "Gaia"
TES_DIR = ROOT / "run" / "tess" / "data" / "Gaia_cdpp_v1"

FLUX_WIN = (0.1, 1e4)      # script-44 science window (insolation, S_earth)
RAD_WIN = (0.6, 2.2)       # script-44 science window (radius, R_earth)
TARGET = 8000              # transiting-in-window planets wanted per type per mission
START_INDEX = 8001         # first new universe index (do not overlap 0..9)


# ── Counting (mirrors script 44) ──────────────────────────────────────────────

def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    if np.issubdtype(s.dtype, np.number):
        return s.fillna(0).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def _stype1(df: pd.DataFrame) -> pd.Series:
    return df["stype"].astype(str).str.strip().str.upper().str[0]


def count_transiting(df: pd.DataFrame, mission: str, stype: str) -> int:
    f = pd.to_numeric(df["flux_p"], errors="coerce")
    r = pd.to_numeric(df["radius_p"], errors="coerce")
    win = ((f > FLUX_WIN[0]) & (f < FLUX_WIN[1])
           & (r > RAD_WIN[0]) & (r < RAD_WIN[1]) & (_stype1(df) == stype))
    if mission == "kepler":
        tr = _as_bool(df["transiting_geometric"])
    else:
        tr = _as_bool(df["tess_observed"]) & _as_bool(df["tess_transiting_geometric"])
    return int((win & tr).sum())


# ── P-Pop generation (one spectral type, one seeded universe) ─────────────────

def generate_typed_universe(stype: str, seed: int, max_stars: int | None = None) -> pd.DataFrame:
    """Generate ONE P-Pop universe over only the `stype` stars of the Gaia-60pc
    catalogue (current config: Bergsten2022 FGK, Chen2017, He2019, ...).
    Returns the standardized P-Pop catalogue (pre-detection)."""
    ppop = PPop(rng=np.random.default_rng(seed))
    ppop.StarCatalog = gaia

    sysgen = SystemGenerator.SystemGenerator(
        ppop.StarCatalog, ppop.StypeToModel, ppop.ScalingModel, ppop.MassModel,
        ppop.EccentricityModel, ppop.StabilityModel, ppop.OrbitModel,
        ppop.AlbedoModel, ppop.ExozodiModel, ppop.Stypes, ppop.Dist_range,
        ppop.Dec_range, ppop.Scenario, ppop.SummaryPlots, 1, ppop.FigDir,
        ppop.block, rng=ppop.rng)

    sc = sysgen.StarCatalog.SC
    st = np.array([(s.decode("ascii", "ignore") if isinstance(s, bytes) else str(s)).strip()[:1].upper()
                   for s in sc["Stype"]])
    sysgen.StarCatalog.SC = sc[st == stype]
    if max_stars is not None:
        sysgen.StarCatalog.SC = sysgen.StarCatalog.SC[:max_stars]
    n_stars = len(sysgen.StarCatalog.SC)
    if n_stars == 0:
        raise RuntimeError(f"No {stype} stars after filtering the Gaia catalogue")

    df = sysgen.SimulateUniverses("gk", Nuniverses=1, write_to_file=False)
    ppop.catalog_from_ppop(df=df)
    return ppop.catalog


def detect_and_save(cat: pd.DataFrame, idx: int, kep_dir: Path, tes_dir: Path):
    """Run BOTH detectors on the shared P-Pop catalogue and save both files."""
    kep = KeplerData(cat.copy())
    kep.determine_detectable()
    kdf = kep.catalog
    kdf.to_csv(kep_dir / f"kepler_catalog_{idx}.csv", index=False)

    tkw = _build_tess_kwargs(None, run_seed=idx)
    tes = TESSData(cat.copy(), source="ppop", **tkw)
    tes.determine_detectable()
    tdf = tes.catalog
    tdf.to_csv(tes_dir / f"tess_catalog_{idx}.csv", index=False)
    return kdf, tdf


# ── Bookkeeping over existing catalogs ────────────────────────────────────────

_USECOLS = ["stype", "flux_p", "radius_p", "transiting_geometric",
            "tess_observed", "tess_transiting_geometric"]


def _read_min(path: Path) -> pd.DataFrame:
    have = pd.read_csv(path, nrows=0).columns
    cols = [c for c in _USECOLS if c in have]
    return pd.read_csv(path, usecols=cols)


def scan_totals(kep_dir: Path, tes_dir: Path):
    """Current transiting-in-window totals across ALL existing catalogs, and the
    next free universe index (>= START_INDEX)."""
    totals = {("kepler", "G"): 0, ("kepler", "K"): 0,
              ("tess", "G"): 0, ("tess", "K"): 0}
    next_idx = START_INDEX
    for mission, directory, stem in (("kepler", kep_dir, "kepler_catalog"),
                                     ("tess", tes_dir, "tess_catalog")):
        for f in sorted(directory.glob(f"{stem}_*.csv")):
            try:
                idx = int(f.stem.rsplit("_", 1)[1])
            except ValueError:
                continue
            d = _read_min(f)
            for st in ("G", "K"):
                totals[(mission, st)] += count_transiting(d, mission, st)
            if idx >= START_INDEX:
                next_idx = max(next_idx, idx + 1)
    return totals, next_idx


# ── Main driver ───────────────────────────────────────────────────────────────

def run(types, target, kep_dir, tes_dir, max_stars=None, max_universes=None):
    kep_dir.mkdir(parents=True, exist_ok=True)
    tes_dir.mkdir(parents=True, exist_ok=True)

    totals, idx = scan_totals(kep_dir, tes_dir)
    print("Current transiting-in-window totals (all existing catalogs):")
    for st in ("G", "K"):
        print(f"  {st}:  Kepler={totals[('kepler', st)]:>6}   TESS={totals[('tess', st)]:>6}"
              f"   (target {target})")
    print(f"Next free universe index: {idx}\n")

    made = 0
    for st in types:
        need_k = max(0, target - totals[("kepler", st)])
        need_t = max(0, target - totals[("tess", st)])
        if need_k == 0 and need_t == 0:
            print(f"[{st}] already at target for both missions — skipping.")
            continue
        print(f"[{st}] deficit: Kepler {need_k}, TESS {need_t} -> generating {st}-only universes")
        while min(totals[("kepler", st)], totals[("tess", st)]) < target:
            if max_universes is not None and made >= max_universes:
                print(f"  reached --max-universes={max_universes}; stopping early.")
                return totals
            t0 = time.time()
            cat = generate_typed_universe(st, seed=idx, max_stars=max_stars)
            kdf, tdf = detect_and_save(cat, idx, kep_dir, tes_dir)
            gk = count_transiting(kdf, "kepler", st)
            gt = count_transiting(tdf, "tess", st)
            totals[("kepler", st)] += gk
            totals[("tess", st)] += gt
            made += 1
            print(f"  uni {idx} [{st}-only, {len(cat):,} planets]: "
                  f"+{gk} Kep / +{gt} TESS  ->  "
                  f"Kep={totals[('kepler', st)]} TESS={totals[('tess', st)]}  "
                  f"({time.time() - t0:.0f}s)")
            idx += 1
    print("\nDone. Final transiting-in-window totals:")
    for st in ("G", "K"):
        print(f"  {st}:  Kepler={totals[('kepler', st)]}   TESS={totals[('tess', st)]}")
    return totals


def smoke_test():
    """Fast self-test: 1 tiny G + 1 tiny K universe into temp dirs; verify the
    full path (generation -> both detectors -> required columns -> counting)."""
    kep_dir = ROOT / "run" / "kepler" / "data" / "_smoke_gk"
    tes_dir = ROOT / "run" / "tess" / "data" / "_smoke_gk"
    for d in (kep_dir, tes_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    print("=== SMOKE TEST (temp dirs, ~250 stars/type) ===")
    try:
        for st in ("G", "K"):
            idx = START_INDEX
            cat = generate_typed_universe(st, seed=idx, max_stars=250)
            kdf, tdf = detect_and_save(cat, idx, kep_dir, tes_dir)
            need_kep = ["transiting_geometric", "stype", "radius_p", "flux_p", "kepler_mes"]
            need_tes = ["tess_transiting_geometric", "tess_observed", "tess_snr", "stype"]
            miss_k = [c for c in need_kep if c not in kdf.columns]
            miss_t = [c for c in need_tes if c not in tdf.columns]
            assert not miss_k, f"Kepler catalog missing {miss_k}"
            assert not miss_t, f"TESS catalog missing {miss_t}"
            print(f"  [{st}] planets={len(cat):,}  "
                  f"kep_transiting_inwin={count_transiting(kdf, 'kepler', st)}  "
                  f"tess_transiting_inwin={count_transiting(tdf, 'tess', st)}  "
                  f"files OK, columns OK")
        print("SMOKE TEST PASSED.")
    finally:
        for d in (kep_dir, tes_dir):
            if d.exists():
                shutil.rmtree(d)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--types", default="G,K",
                    help="spectral types to top up, in order (default: G,K)")
    ap.add_argument("--target", type=int, default=TARGET,
                    help="transiting-in-window target per type per mission")
    ap.add_argument("--max-stars", type=int, default=None,
                    help="cap stars per universe (testing/speed)")
    ap.add_argument("--max-universes", type=int, default=None,
                    help="stop after generating this many universes (testing)")
    ap.add_argument("--smoke", action="store_true", help="run fast self-test and exit")
    args = ap.parse_args()

    if args.smoke:
        smoke_test()
        return

    types = [t.strip().upper() for t in args.types.split(",") if t.strip()]
    run(types, args.target, KEP_DIR, TES_DIR,
        max_stars=args.max_stars, max_universes=args.max_universes)


if __name__ == "__main__":
    main()

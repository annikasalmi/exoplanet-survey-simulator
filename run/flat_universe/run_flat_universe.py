"""Flat A/B pipeline: builds universes A and B, runs Kepler/TESS/RV detection, caches the result."""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from science.physics import is_super_earth
from science.populations.universes.flat_curves import flat_radii_curves
from science.telescopes.detection import run_rv_best
from science.telescopes.kepler.detection_model import KeplerData
from science.telescopes.tess.detection_model import TESSData
from tools.paths import FLAT_UNIVERSE_DATA_DIR

FLAT_CACHE_DIR = Path(FLAT_UNIVERSE_DATA_DIR)
FLAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
N_PLANETS = 150_000
CACHE_NAMES = {"A": "flat_subneptunes_only", "B": "flat_superearths_subneptunes"}


def _get_or_generate_universes(seed, n_planets, run_anew):
    """Load cached A and B, or build them. A is detected B minus its super-Earths."""
    cache = {u: FLAT_CACHE_DIR / f"{name}_seed{seed}_n{n_planets}.csv" for u, name in CACHE_NAMES.items()}

    if not run_anew and all(f.exists() for f in cache.values()):
        for f in cache.values():
            print(f"  Loading cached {f.name}")
        return {u: pd.read_csv(f) for u, f in cache.items()}

    print(f"  Generating universe B (seed={seed}, n={n_planets:,})...")
    b = flat_radii_curves(
        n_planets, seed=seed, variant="superearths_supneptunes"
    )
    b['kepler_detected'] = KeplerData(
        b.copy(), source="ppop"
    ).determine_detectable()['detected']
    b['tess_detected'] = TESSData(
        b.copy(), source="ppop", use_cdpp_tables=False
    ).determine_detectable()['detected']
    b['rv_detected'] = run_rv_best(b, mag_target=12.0)['detected']
    a = b[~is_super_earth(b['mass_p'], b['radius_p'])].reset_index(drop=True)
    print(f"    B: {len(b):,} planets; A: {len(a):,} after dropping super-Earths")

    universes = {"A": a.assign(universe_type="A"), "B": b.assign(universe_type="B")}
    for u, df in universes.items():
        df.to_csv(cache[u], index=False)
        print(f"  Cached: {cache[u].name}")
    return universes


def main(
    seed=SEED,
    n_planets=N_PLANETS,
    parallel=False,
    nruns=np.arange(1),
    star_catalog=None,
    run_anew=True,
):
    """A and B concatenated with detection, universe_type and run columns. run_anew=False reuses the
    cache. parallel, nruns and star_catalog are unused; they match run_sim's pipeline signature."""
    start = time.time()
    print(f"Flat universe: seed={seed}, n_planets={n_planets:,}")

    universes = _get_or_generate_universes(seed=seed, n_planets=n_planets, run_anew=run_anew)
    results = [universes[u].assign(run=0) for u in ("A", "B")]

    df_concat = pd.concat(results, ignore_index=True)

    elapsed = time.time() - start
    print(f"\nFlat universe complete in {elapsed:.1f}s")
    print(f"  Total: {len(df_concat):,} planets (A {len(results[0]):,}, B {len(results[1]):,})")

    return df_concat


if __name__ == "__main__":
    df = main()
    print("\nColumns:", df.columns.tolist())

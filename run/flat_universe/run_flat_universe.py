"""Flat A/B pipeline: builds universes A and B, runs Kepler/TESS/RV detection, caches the result."""

from __future__ import annotations

import time
import pandas as pd
from pathlib import Path

from science.populations.universes.flat_curves import flat_curves, is_super_earth
from science.telescopes.kepler.detection_model import KeplerData
from science.telescopes.tess.detection_model import TESSData
from science.telescopes.detection import run_rv_best
from tools.paths import FLAT_UNIVERSE_DATA_DIR

FLAT_CACHE_DIR = Path(FLAT_UNIVERSE_DATA_DIR)

SEED = 0
N_PLANETS = 150_000
CACHE_NAMES = {"A": "flat_subneptunes_only", "B": "flat_superearths_subneptunes"}


def main(seed=SEED, n_planets=N_PLANETS, run_anew=True):
    """Generate/detect B once, derive A, and cache both. run_anew=False reuses the cache."""
    start = time.time()
    FLAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = {u: FLAT_CACHE_DIR / f"{name}_seed{seed}_n{n_planets}.csv" for u, name in CACHE_NAMES.items()}

    if not run_anew and all(f.exists() for f in cache.values()):
        for f in cache.values():
            print(f"  Loading cached {f.name}")
        universes = {u: pd.read_csv(f) for u, f in cache.items()}
    else:
        print(f"  Generating universe B (seed={seed}, n={n_planets:,})...")
        b = flat_curves(n_planets, seed=seed)
        b['kepler_detected'] = KeplerData(b.copy(), source="ppop").determine_detectable()['detected']
        b['tess_detected'] = TESSData(b.copy(), source="ppop", use_cdpp_tables=False).determine_detectable()['detected']
        b['rv_detected'] = run_rv_best(b, mag_target=12.0)['detected']
        a = b[~is_super_earth(b['mass_p'], b['radius_p'])].reset_index(drop=True)
        universes = {"A": a.assign(universe_type="A"), "B": b.assign(universe_type="B")}
        for u, df in universes.items():
            df.to_csv(cache[u], index=False)
            print(f"  Cached: {cache[u].name}")
    results = [universes[u].assign(run=0) for u in ("A", "B")]

    df_concat = pd.concat(results, ignore_index=True)

    elapsed = time.time() - start
    print(f"\nFlat universe complete in {elapsed:.1f}s")
    print(f"  Total: {len(df_concat):,} planets (A {len(results[0]):,}, B {len(results[1]):,})")

    return df_concat


if __name__ == "__main__":
    df = main()
    print("\nColumns:", df.columns.tolist())

"""Flat A/B pipeline: generates flat catalogs (A drops rocky M > 2, B keeps all), runs
Kepler/TESS/RV detection, and caches by seed and size so plots rerun fast.
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from run.flat_universe.uniform_generator import generate_flat_catalog
from run.ppop.flat_detect import run_kepler, run_tess, run_rv_best
from tools.paths import FLAT_UNIVERSE_DATA_DIR

FLAT_CACHE_DIR = Path(FLAT_UNIVERSE_DATA_DIR)
FLAT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Hongyi's original flat universe parameters (for reproducibility)
HONGYI_CONFIGS = {
    'flat_rocky_mr_vs_nasa': {'seed': 0, 'n_planets': 150_000},
    'flat_transit_rv_3x3_G': {'seed': 75, 'n_planets': 3_000_000},
    'flat_transit_rv_3x3_K': {'seed': 76, 'n_planets': 2_500_000},
    'flat_transit_rv_3x3_M': {'seed': 77, 'n_planets': 1_500_000},
    'likelihood_ratio_catalog': {'seed': 0, 'n_planets': 1_000_000},
}


def _get_or_generate_universe(seed=0, n_planets=150000, universe_type='A'):
    """Load the cached flat universe or generate it. universe_type: 'A' (drop rocky M>2) or 'B' (all)."""
    cache_file = FLAT_CACHE_DIR / f"flat_universe_{universe_type}_seed{seed}_n{n_planets}.csv"

    if cache_file.exists():
        print(f"  Loading cached {universe_type}: {cache_file.name}")
        return pd.read_csv(cache_file)

    print(f"  Generating universe {universe_type} (seed={seed}, n={n_planets:,})...")
    df = generate_flat_catalog(n_planets=n_planets, seed=seed)

    # Universe A: drop rocky planets with TRUE mass > 2 M_earth
    if universe_type == 'A':
        df = df[df['mass_p'] <= 2.0].copy()
        print(f"    → {len(df):,} planets after dropping M>2")

    df['kepler_detected'] = run_kepler(df)['detected']
    df['tess_detected'] = run_tess(df)['detected']
    df['rv_detected'] = run_rv_best(df, mag_target=12.0)['detected']
    df['universe_type'] = universe_type

    # Cache
    df.to_csv(cache_file, index=False)
    print(f"  Cached: {cache_file.name}")

    return df


def main(
    seed=0,
    n_planets=150000,
    parallel=False,
    nruns=np.arange(1),
    star_catalog=None,
    run_anew=True,
):
    """Flat-universe (A and B) detection pipeline; returns both concatenated with kepler_/tess_/rv_detected,
    universe_type and run columns. run_anew=True ignores the cache. parallel, nruns and star_catalog are
    unused, accepted so run_sim can call this like the telescope pipelines.
    """
    start = time.time()
    print(f"Flat universe: seed={seed}, n_planets={n_planets:,}")

    results = []
    for universe_type in ['A', 'B']:
        print(f"\nUniverse {universe_type}:")
        df = _get_or_generate_universe(
            seed=seed,
            n_planets=n_planets,
            universe_type=universe_type
        )
        df['run'] = 0
        results.append(df)

    df_concat = pd.concat(results, ignore_index=True)

    elapsed = time.time() - start
    print(f"\nFlat universe complete in {elapsed:.1f}s")
    print(f"  Total: {len(df_concat):,} planets ({len(df_concat)//2:,} per universe)")

    return df_concat


if __name__ == "__main__":
    df = main(seed=0, n_planets=150000)
    print("\nColumns:", df.columns.tolist())

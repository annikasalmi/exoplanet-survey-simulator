"""
run_flat_universe.py — flat A/B synthetic planet detection pipeline.

Generates fully-flat (parameter-independent) synthetic catalogs (A=drop rocky
M>2, B=all rocky) and runs Kepler/TESS/RV detection for detector sensitivity analysis.

Caches results by seed/size to enable fast reruns of plots.
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
    'two_universe_puffy_overlap': {'seed': 0, 'n_planets': 300_000},
    'likelihood_ratio_catalog': {'seed': 0, 'n_planets': 1_000_000},
    'mc_mr_density_priors': {'seed': 1},
}


def _get_or_generate_universe(seed=0, n_planets=150000, universe_type='A'):
    """
    Get cached or generate flat universe.

    universe_type: 'A' (drop rocky M>2) or 'B' (all rocky)
    """
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
    """
    Flat-universe (A and B) detection pipeline.

    Args:
        seed: Random seed for catalog generation
        n_planets: Number of planets per universe
        run_anew: If True, always regenerate (ignore cache)
        parallel, nruns, star_catalog: unused. The flat universe draws its own
            stars, so there is no catalog to pick; accepted so run_sim can call
            this the same way it calls the telescope pipelines.

    Returns:
        DataFrame with A and B concatenated, columns: all planet properties +
        kepler_detected, tess_detected, rv_detected, universe_type, run
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

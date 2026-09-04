#!/usr/bin/env python
"""
generate_flat_universes.py — Generate and cache flat universe test populations.

Different science use cases need different flat-universe configs. This script
generates the standard configurations used by Tier 2/3 analysis scripts.

Configurations:
  - rocky_mr_powerlaw: 150k planets, powerlaw M-R with scatter, seed=0
  - transit_rv_by_stype: 3M planets, independent mass, seeds per spectral type (F:75, G:76, K:77, M:78)
  - puffy_overlap_base: 300k planets, independent mass, seed=0
  - nominal_150k: 150k planets, independent mass, seed=0
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run.flat_universe.uniform_generator import get_or_build_catalog, UNIFORM_OUT_DIR


CONFIGS = {
    "rocky_mr_powerlaw": {
        "cache_path": os.path.join(UNIFORM_OUT_DIR, "flat_150k_seed0_powerlaw.csv"),
        "n_planets": 150000,
        "seed": 0,
        "mass_model": "powerlaw",
        "mass_scatter_dex": 0.1,
        "mr_C": 2.5,
        "mr_beta": 0.6,
    },
    "transit_rv_by_stype": {
        "cache_path": os.path.join(UNIFORM_OUT_DIR, "flat_3M_seed75-78_independent.csv"),
        "n_planets": 3_000_000,
        "seed": 0,  # combined seed; actual seeds per stype in script
        "mass_model": "independent",
    },
    "puffy_overlap_base": {
        "cache_path": os.path.join(UNIFORM_OUT_DIR, "flat_300k_seed0_independent.csv"),
        "n_planets": 300_000,
        "seed": 0,
        "mass_model": "independent",
    },
    "nominal_150k": {
        "cache_path": os.path.join(UNIFORM_OUT_DIR, "flat_150k_seed0_independent.csv"),
        "n_planets": 150_000,
        "seed": 0,
        "mass_model": "independent",
    },
}


def generate_config(config_name: str, rebuild: bool = False):
    """Generate a single flat universe configuration."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Options: {list(CONFIGS.keys())}")

    cfg = CONFIGS[config_name]
    cache_path = cfg.pop("cache_path")

    print(f"Generating config '{config_name}'...")
    if (not rebuild) and os.path.exists(cache_path):
        print(f"  Already cached at {cache_path}")
        return

    catalog = get_or_build_catalog(cache_path, rebuild=True, **cfg)
    print(f"  Generated {len(catalog)} planets → {cache_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate and cache flat universe test populations."
    )
    parser.add_argument(
        "configs",
        nargs="*",
        default=["nominal_150k"],
        help=f"Config names to generate. Choices: {list(CONFIGS.keys())} (default: nominal_150k)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all configs",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild even if cached",
    )

    args = parser.parse_args()

    os.makedirs(UNIFORM_OUT_DIR, exist_ok=True)

    if args.all:
        configs_to_gen = list(CONFIGS.keys())
    else:
        configs_to_gen = args.configs

    for config_name in configs_to_gen:
        try:
            generate_config(config_name, rebuild=args.rebuild)
        except ValueError as e:
            print(f"  ERROR: {e}")
            sys.exit(1)

    print("Done.")


if __name__ == "__main__":
    main()

"""Backward-compatibility wrapper — use run.flat_universe instead."""
from run.flat_universe.uniform_generator import (
    generate_flat_catalog,
    get_or_build_catalog,
    UNIFORM_OUT_DIR,
)

__all__ = ["generate_flat_catalog", "get_or_build_catalog", "UNIFORM_OUT_DIR"]

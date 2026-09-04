#!/usr/bin/env python
"""End-to-end check: run_flat_universe -> cache -> plot_flat_universe."""

import pandas as pd

from run.flat_universe.run_flat_universe import main as run_flat, HONGYI_CONFIGS
from output.plots.plot_flat_universe import plot_flat_universe

cfg = HONGYI_CONFIGS["likelihood_ratio_catalog"]

df = run_flat(**cfg)
assert set(df["universe_type"]) == {"A", "B"}
assert {"kepler_detected", "tess_detected", "rv_detected"} <= set(df.columns)
assert (df[df.universe_type == "A"].mass_p <= 2.0).all(), "universe A must drop M>2"
assert len(df[df.universe_type == "A"]) < len(df[df.universe_type == "B"])
print(f"[ok] generated {len(df):,} planets")

pd.testing.assert_frame_equal(df, run_flat(**cfg), check_dtype=False)
print("[ok] cache reload is identical")

plot_flat_universe(df)
print("[ok] plots generated")

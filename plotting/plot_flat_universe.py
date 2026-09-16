"""Plots driven by the flat-universe simulation alone. Plots that also need P-Pop,
Kepler or TESS runs live in plotting/scripts/analysis/multi/.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

from plotting import likelihood_ratio_plotter


def plot_flat_universe(df, nruns=1, use_multiprocessing=False, **kwargs):
    """Make the flat-universe plots from run_flat_universe output. df needs radius_p,
    mass_p, flux_p, teff_s, kepler_detected, rv_detected and universe_type.
    """
    print(f"\nPlotting flat universe ({len(df):,} planets, "
          f"universes {sorted(df['universe_type'].unique())})")

    likelihood_ratio_plotter.main(df)
    print("Flat universe plotting complete.\n")

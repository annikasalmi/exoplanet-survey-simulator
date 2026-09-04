"""
plot_flat_universe.py — plots driven by the flat-universe simulation.

Only plots that depend on the flat universe alone live here. Anything needing a
second simulation (P-Pop, Kepler, TESS) lives in output/plots/scripts/analysis/multi/.
"""

from __future__ import annotations

import matplotlib
matplotlib.use('Agg')

from output.plots import likelihood_ratio_plotter


def plot_flat_universe(df, nruns=1, use_multiprocessing=False, **kwargs):
    """
    Generate flat-universe plots from data produced by run_flat_universe.

    df must carry: radius_p, mass_p, flux_p, teff_s, kepler_detected,
    rv_detected, universe_type.
    """
    print(f"\nPlotting flat universe ({len(df):,} planets, "
          f"universes {sorted(df['universe_type'].unique())})")

    try:
        likelihood_ratio_plotter.main(df)
        print("[ok] likelihood_ratio_catalog")
    except Exception as e:
        print(f"[fail] likelihood_ratio_catalog: {e}")

    print("Flat universe plotting complete.\n")

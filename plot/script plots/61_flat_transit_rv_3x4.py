"""
61_flat_transit_rv_3x4.py

FLAT-universe analog of script 53. Same 3x4 R-insolation format and styling
(rows = Transit-test / RV-test / Transit+RV joint, cols = F G K M; viridis pass-
fraction background; white contours; rocky NASA overlay colored by facility with
white outlines; black 90% upper-bound line; cold red window), but every
background pass-fraction is computed on the fully-FLAT control catalogue
(run/ppop/uniform_generator.py) instead of the Gaia-60pc P-Pop TESS catalogs.

    Row 0  Transit-test       P(transit detected | transiting)   -- uses TESS
    Row 1  RV-test (alone)    P(mass RV-measurable | transiting)
    Row 2  Transit + RV       P(both | transiting) = rocky-sample selection fn

As in script 53 the flat background is restricted to planets BELOW the rocky
threshold curve (the same unshifted silicate curve, Hongyi-silicon.ddat), because
at fixed radius the RV signal depends on mass: the question is specifically
whether ROCKY planets of that radius are confirmable. The transit row uses the
TESS detector (per the request), matching script 53's transit row.

Because the flat catalogue draws parameters uniformly and independently, the
background is the DETECTOR selection function free of any occurrence-rate prior:
if the cold red window is bright on the transit row but dark on the joint row,
the missing cold rocky planets are an RV selection effect, not a real radius
limit -- read exactly as in script 53, but on the unbiased control population.

Run from repo root:
    python scripts/61_flat_transit_rv_3x4.py
    python scripts/61_flat_transit_rv_3x4.py --instrument HARPS --n-planets 600000 --rebuild
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse scripts 44 (threshold / NASA / grid / axis) and 53 (RV test, warm
# overlay, OLS+shift 90% line, per-panel renderer) so styling matches exactly.
S44 = _load_module("s44", ROOT / "plot" / "script plots" / "56_kepler_tess_rocky_fgkm_gaia60pc.py")
S53 = _load_module("s53", ROOT / "plot" / "script plots" / "58_rv_rocky_fgkm_test.py")

from run.ppop.uniform_generator import get_or_build_catalog, UNIFORM_OUT_DIR
from run.ppop.flat_detect import run_tess

STAR_ORDER         = S44.STAR_ORDER
INSOLATION_BINS    = S44.INSOLATION_BINS
PLANET_RADIUS_BINS = S44.PLANET_RADIUS_BINS
INSOLATION_LIMITS  = S44.INSOLATION_LIMITS
RADIUS_LIMITS      = S44.RADIUS_LIMITS
COLD_INSOLATION    = S44.COLD_INSOLATION
TESS_SNR_THRESHOLD = S44.TESS_SNR_THRESHOLD
CMAP_DETECTED      = S44.CMAP_DETECTED

N_PLANETS_DEFAULT = 1_000_000
OUT_DIR = ROOT / "my_outputs" / "61_flat_transit_rv_3x4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cache_path_for(n_planets: int, seed: int) -> str:
    """Reuse script 37's shared 300k/seed-0 cache; otherwise write a distinct
    file so a larger draw never clobbers the shared flat_catalog.csv that scripts
    37 / 56 / 72 / 73 depend on."""
    if n_planets == 300_000 and seed == 0:
        return os.path.join(UNIFORM_OUT_DIR, "flat_catalog.csv")
    return os.path.join(UNIFORM_OUT_DIR, f"flat_catalog_n{n_planets}_s{seed}.csv")

# Same three tests / row order as script 53.
ROWS = [
    ("Transit-test",      lambda p: p["detected_case"].to_numpy()),
    ("RV-test",           lambda p: p["rv_ok"].to_numpy()),
    ("Transit + RV-test", lambda p: p["detected_case"].to_numpy() & p["rv_ok"].to_numpy()),
]


# ── Flat-universe TESS frame (44/53 convention) ──────────────────────────────

def prepare_flat_tess(det_df: pd.DataFrame) -> pd.DataFrame:
    """Flat TESS detector output -> detected_case / denominator_case in the
    science window, keeping mass_p for the rocky cut and the RV test."""
    df = S44.add_stype_clean(det_df.copy())
    for col in ["flux_p", "radius_p", "mass_p", "tess_snr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["tess_observed", "tess_transiting_geometric",
                "tess_star_bright_enough", "tess_enough_transits"]:
        if col in df.columns:
            df[col] = S44.as_bool(df[col])
        else:
            df[col] = True

    df = df.dropna(subset=["flux_p", "radius_p", "mass_p", "tess_snr"]).copy()
    df = S44.restrict_science_window(df)

    detected = S44.as_bool(df["detected"])
    denom = df["tess_observed"] & df["tess_transiting_geometric"]
    df["detected_case"] = detected & denom
    df["denominator_case"] = denom
    print(f"  TESS flat in window: {len(df):,} rows  "
          f"({int(denom.sum()):,} transiting, {int((detected & denom).sum()):,} detected)")
    return df


# ── 3x4 figure (mirrors S53.plot_rv_test, flat background) ───────────────────

def plot_flat_3x4(ppop: pd.DataFrame, rocky_win: pd.DataFrame,
                  color_map: dict, major: list[str], counts, args,
                  out_path: Path | None = None, title_extra: str = "") -> Path:
    fig, axes = plt.subplots(3, 4, figsize=(24, 15.5), sharex=True, sharey=True,
                             constrained_layout=True)
    mesh = None

    print("\nCold red-window false-negative probabilities (I<10, radius above 90% fit):")
    for i, (test_name, numerator_fn) in enumerate(ROWS):
        for j, stype in enumerate(STAR_ORDER):
            ax = axes[i, j]
            p = ppop[ppop["stype_clean"] == stype].copy()
            r = rocky_win[rocky_win["stype_clean"] == stype].copy()

            # Inlined from S53._draw_panel so the subtitle can report the FLAT
            # universe's in-window count (headline) + transiting subset (parens).
            n_window = len(p)                              # rocky flat planets in the science window
            n_trans  = int(p["denominator_case"].sum())   # transiting subset (fills the bins)
            numerator = numerator_fn(p)
            grid, _ = S44.fraction_grid(p, pd.Series(numerator, index=p.index),
                                        p["denominator_case"])
            if np.isfinite(grid).any():
                mesh = ax.pcolormesh(INSOLATION_BINS, PLANET_RADIUS_BINS, grid,
                                     shading="auto", vmin=0, vmax=1, cmap=CMAP_DETECTED)
                S44._add_contours(ax, grid)
            else:
                ax.text(0.5, 0.5, "no flat data", transform=ax.transAxes,
                        ha="center", va="center", color="0.5")

            S53._overlay_rocky_outlined(ax, r, color_map, major)

            if stype in ("G", "K", "M"):
                fit = S53.draw_90pct_line_local(ax, r["flux_p"].values, r["radius_p"].values,
                                                color="black", lw=2.0, x_extent=INSOLATION_LIMITS)
                if fit is not None:
                    slope, intercept = fit
                    xreg = np.logspace(np.log10(INSOLATION_LIMITS[0]),
                                       np.log10(COLD_INSOLATION), 100)
                    ax.fill_between(xreg, S44.line_radius(slope, intercept, xreg),
                                    RADIUS_LIMITS[1], color="red", alpha=0.12,
                                    zorder=1.5, lw=0)
                    fn, n_denom, n_missed = S53.window_stats(p, slope, intercept, numerator)
                    if fn is not None:
                        ax.text(0.03, 0.97,
                                f"P(false neg | I<10, R>fit) = {fn:.0%}\n"
                                f"({n_missed}/{n_denom} flat missed)",
                                transform=ax.transAxes, va="top", ha="left", fontsize=7,
                                color="darkred",
                                bbox=dict(boxstyle="round", fc="white", ec="darkred", alpha=0.85),
                                zorder=8)
                        print(f"  {test_name:18s} {stype}: {fn:.1%}  ({n_missed}/{n_denom})")

            S44._setup_axis(
                ax,
                f"{test_name} — {stype} stars\n"
                f"N_flat rocky in window = {n_window:,}  ({n_trans:,} transiting)   "
                f"·   N_rocky_NASA = {len(r)}",
                show_xlabel=(i == 2), show_ylabel=(j == 0),
            )

    fig.legend(handles=S53._legend_handles(rocky_win, color_map, major, counts),
               loc="outside lower center", ncol=4, fontsize=9, framealpha=0.9,
               title="Rocky NASA planets — colored by discovery facility (telescope/mission)",
               title_fontsize=10)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(),
                     label="Pass fraction (rocky flat planets)", shrink=0.6, pad=0.01)
    fig.suptitle(
        (f"{title_extra}\n" if title_extra else "") +
        "Transit-test (top) · RV-test alone (middle) · Transit + RV joint (bottom) — "
        "FLAT-universe rocky selection function, FGKM\n"
        "Background = rocky FLAT control planets (below the unshifted silicate threshold, "
        "Hongyi-silicon.ddat); transit row = TESS; "
        f"RV-test = mass RV-measurable ({args.instrument}: K/σ_K ≥ {args.snr_threshold}, "
        f"N={args.n_obs}, band-mag ≤ {args.mag_target})\n"
        "If the cold red window is bright on top but dark in the joint row, the missing cold "
        "rocky planets are a selection effect — not a real radius limit",
        fontsize=13,
    )

    out = Path(out_path) if out_path is not None else OUT_DIR / f"flat_transit_rv_3x4_{args.instrument.lower()}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved combined 3x4 flat figure: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-planets", type=int, default=N_PLANETS_DEFAULT,
                    help="Flat-catalogue size to build/load (default: 300000).")
    ap.add_argument("--rebuild", action="store_true",
                    help="Rebuild the flat catalogue cache instead of loading it.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--instrument", default="best", help="HARPS | NIRPS | best")
    ap.add_argument("--snr-threshold", type=float, default=5.0)
    ap.add_argument("--n-obs", type=int, default=100)
    ap.add_argument("--mag-target", type=float, default=12.0)
    args = ap.parse_args()

    cache_csv = cache_path_for(args.n_planets, args.seed)

    print("=" * 70)
    print("61_flat_transit_rv_3x4.py — flat-universe companion to script 53")
    print("=" * 70)
    print(f"Project root : {ROOT}")
    print(f"Flat cache   : {cache_csv}")
    print(f"Output dir   : {OUT_DIR}")
    print()

    # NASA rocky overlay + threshold (identical to scripts 44 / 53).
    m_ref, r_ref = S44.load_rocky_reference_curve()
    shift = S44.compute_rocky_threshold_shift(m_ref, r_ref)
    print()
    nasa_all, rocky = S44.load_and_filter_nasa(m_ref, r_ref, shift)
    rocky_win = S44.restrict_to_window(rocky)
    print(f"Rocky NASA planets in window: {len(rocky_win):,}")
    color_map, major, counts = S53.build_warm_facility_styles(rocky_win)
    print()

    # Flat control catalogue -> TESS detector -> rocky cut -> RV test.
    catalog = get_or_build_catalog(cache_csv, rebuild=args.rebuild,
                                   n_planets=args.n_planets, seed=args.seed)
    print(f"--> Running TESS detector on {len(catalog):,} flat planets...")
    ppop = prepare_flat_tess(run_tess(catalog))
    ppop = S53.restrict_ppop_to_rocky(ppop, m_ref, r_ref, shift)
    ppop = S53.add_rv_ok(ppop, args.instrument, args.snr_threshold, args.n_obs, args.mag_target)

    plot_flat_3x4(ppop, rocky_win, color_map, major, counts, args)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

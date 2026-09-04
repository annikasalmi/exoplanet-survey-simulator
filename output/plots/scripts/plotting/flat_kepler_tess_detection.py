"""
60_flat_kepler_tess_detection.py

FLAT-universe analog of script 44. Same 2x4 R-insolation format and styling
(rows = Kepler / TESS, cols = F G K M; viridis detected-fraction background;
white contours; rocky NASA overlay colored by discovery facility; black 90%
upper-bound line; cold red false-negative window), but the detected-fraction
BACKGROUND is computed on the fully-FLAT control catalogue
(run/ppop/uniform_generator.py) instead of the Gaia-60pc P-Pop universes.

Why a flat background: the P-Pop background in script 44 is
parent_occurrence x detection_probability, so the panel mixes the input
occurrence prior with the detector. The flat catalogue draws every planet/star
parameter independently and uniformly, so each (insolation, radius) bin starts
with a roughly equal number of planets and the detected fraction reads off the
DETECTOR's sensitivity directly -- the same quantity script 37 plots as 1-D
curves, here as the full 2-D map under the real rocky planets.

The NASA rocky overlay, the rocky threshold, the 90% upper-bound line and the
cold (I<10) false-negative box are identical to script 44 -- only the colored
background changes.

Run from repo root:
    python scripts/60_flat_kepler_tess_detection.py
    python scripts/60_flat_kepler_tess_detection.py --n-planets 600000 --rebuild
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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


# Reuse script 44's machinery so geometry / styling / rocky threshold / NASA
# loading all match the transit figure exactly.
S44 = _load_module("s44", ROOT / "plot" / "script plots" / "56_kepler_tess_rocky_fgkm_gaia60pc.py")

from run.ppop.uniform_generator import get_or_build_catalog, UNIFORM_OUT_DIR
from run.ppop.flat_detect import run_kepler, run_tess

# Shared constants (identical bins / window / colormap as script 44).
STAR_ORDER         = S44.STAR_ORDER
INSOLATION_BINS    = S44.INSOLATION_BINS
PLANET_RADIUS_BINS = S44.PLANET_RADIUS_BINS
INSOLATION_LIMITS  = S44.INSOLATION_LIMITS
RADIUS_LIMITS      = S44.RADIUS_LIMITS
COLD_INSOLATION    = S44.COLD_INSOLATION
CMAP_DETECTED      = S44.CMAP_DETECTED
OTHER_COLOR        = S44.OTHER_COLOR

N_PLANETS_DEFAULT = 1_000_000
OUT_DIR = ROOT / "output/plots" / "60_flat_kepler_tess_detection"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def cache_path_for(n_planets: int, seed: int) -> str:
    """Reuse script 37's shared 300k/seed-0 cache; otherwise write a distinct
    file so a larger draw never clobbers the shared flat_catalog.csv that scripts
    37 / 56 / 72 / 73 depend on."""
    if n_planets == 300_000 and seed == 0:
        return os.path.join(UNIFORM_OUT_DIR, "flat_catalog.csv")
    return os.path.join(UNIFORM_OUT_DIR, f"flat_catalog_n{n_planets}_s{seed}.csv")


# ── Flat-universe detector frames ─────────────────────────────────────────────

def prepare_flat(det_df: pd.DataFrame, mission: str) -> pd.DataFrame:
    """Take a flat-catalogue detector output and attach detected_case /
    denominator_case in the same convention script 44 uses, restricted to the
    science window."""
    df = S44.add_stype_clean(det_df.copy())
    for col in ["flux_p", "radius_p"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["flux_p", "radius_p"]).copy()
    df = S44.restrict_science_window(df)

    detected = S44.as_bool(df["detected"])
    if mission == "Kepler":
        denom = S44.as_bool(df["transiting_geometric"])
    else:  # TESS
        denom = S44.as_bool(df["tess_observed"]) & S44.as_bool(df["tess_transiting_geometric"])

    # detected already implies transiting; AND with denom for safety so the
    # numerator can never exceed the denominator in any bin.
    df["detected_case"] = detected & denom
    df["denominator_case"] = denom
    print(f"  {mission:6s} flat in window: {len(df):,} rows  "
          f"({int(denom.sum()):,} transiting, {int((detected & denom).sum()):,} detected)")
    return df


# ── 2x4 figure (mirrors S44.plot_combined, flat background) ──────────────────

def plot_combined_flat(kepler: pd.DataFrame, tess: pd.DataFrame,
                       rocky_win: pd.DataFrame, color_map: dict,
                       major: list[str], counts) -> Path:
    n_other = int(len(rocky_win) - sum(counts[f] for f in major))

    rows = [("Kepler", kepler), ("TESS", tess)]
    fig, axes = plt.subplots(2, 4, figsize=(24, 11), sharex=True, sharey=True,
                             constrained_layout=True)
    mesh = None

    print("\nCold-region false-negative probabilities (I<10, radius above 90% fit):")
    for i, (mission, ppop) in enumerate(rows):
        for j, stype in enumerate(STAR_ORDER):
            ax = axes[i, j]
            p = ppop[ppop["stype_clean"] == stype].copy()
            r = rocky_win[rocky_win["stype_clean"] == stype].copy()

            n_window = len(p)                                # flat planets in the science window
            n_trans  = int(p["denominator_case"].sum())      # transiting subset (fills the bins)
            det_grid, _ = S44.fraction_grid(p, p["detected_case"], p["denominator_case"])
            if np.isfinite(det_grid).any():
                mesh = ax.pcolormesh(
                    INSOLATION_BINS, PLANET_RADIUS_BINS, det_grid,
                    shading="auto", vmin=0, vmax=1, cmap=CMAP_DETECTED,
                )
                S44._add_contours(ax, det_grid)
            else:
                ax.text(0.5, 0.5, "no flat data", transform=ax.transAxes,
                        ha="center", va="center", color="0.5")

            S44._overlay_rocky_by_facility(ax, r, color_map, major)

            if stype in ("G", "K", "M"):
                fit = S44.draw_90pct_line(
                    ax, r["flux_p"].values, r["radius_p"].values,
                    color="black", lw=2.0, x_extent=INSOLATION_LIMITS,
                )
                if fit is not None:
                    slope, intercept = fit
                    xreg = np.logspace(np.log10(INSOLATION_LIMITS[0]),
                                       np.log10(COLD_INSOLATION), 100)
                    ax.fill_between(xreg, S44.line_radius(slope, intercept, xreg),
                                    RADIUS_LIMITS[1], color="red", alpha=0.12,
                                    zorder=1.5, lw=0)
                    fn, n_denom, n_missed = S44.false_negative_prob(p, slope, intercept)
                    if fn is not None:
                        ax.text(0.03, 0.97,
                                f"P(false neg | I<10, R>fit) = {fn:.0%}\n"
                                f"({n_missed}/{n_denom} flat missed)",
                                transform=ax.transAxes, va="top", ha="left",
                                fontsize=7, color="darkred",
                                bbox=dict(boxstyle="round", fc="white",
                                          ec="darkred", alpha=0.85), zorder=8)
                        print(f"  {mission:6s} {stype}: "
                              f"P(false neg | I<10, R>fit) = {fn:.1%}  "
                              f"({n_missed}/{n_denom} flat)")

            S44._setup_axis(
                ax,
                f"{mission} — {stype} stars\n"
                f"N_flat in window = {n_window:,}  ({n_trans:,} transiting)   ·   N_rocky = {len(r)}",
                show_xlabel=(i == 1),
                show_ylabel=(j == 0),
            )

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[f],
               markersize=7, label=f"{S44.short_facility(f)}  (N={counts[f]})")
        for f in major
    ]
    if n_other > 0:
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", color=OTHER_COLOR,
                   markersize=7, label=f"Other facilities  (N={n_other})")
        )
    handles += [
        Line2D([0], [0], marker="*", linestyle="", color="gold",
               markeredgecolor="darkred", markersize=12, label="LHS 1140 b (anchor)"),
        Line2D([0], [0], color="black", lw=2.0, ls="--",
               label="90% upper-bound fit (≈90% of planets below; G/K/M)"),
        Patch(facecolor="red", alpha=0.12,
              label="Cold false-negative region (I<10, R>fit)"),
    ]
    fig.legend(
        handles=handles, loc="outside lower center",
        ncol=min(len(handles), 4), fontsize=9, framealpha=0.9,
        title="Rocky NASA planets — colored by discovery facility (telescope/mission)",
        title_fontsize=10,
    )
    fig.colorbar(mesh, ax=axes.ravel().tolist(),
                 label="Detected fraction (of transiting)", shrink=0.6, pad=0.01)
    fig.suptitle(
        "Kepler (top) & TESS (bottom) FGKM — FLAT-universe detected-fraction background "
        "+ rocky PSCompPars overlay\n"
        "Background = detection efficiency on the fully-flat control catalogue "
        "(uniform_generator); rocky threshold = silicate curve (Hongyi-silicon.ddat), unshifted  |  "
        "red error bars = two-sided radius uncertainty",
        fontsize=13,
    )

    out = OUT_DIR / "flat_kepler_tess_2x4_detection.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved combined 2x4 flat figure: {out}")
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
    args = ap.parse_args()

    cache_csv = cache_path_for(args.n_planets, args.seed)

    print("=" * 70)
    print("60_flat_kepler_tess_detection.py — flat-universe companion to script 44")
    print("=" * 70)
    print(f"Project root : {ROOT}")
    print(f"Flat cache   : {cache_csv}")
    print(f"Output dir   : {OUT_DIR}")
    print()

    # NASA rocky overlay + threshold (identical to script 44).
    m_ref, r_ref = S44.load_rocky_reference_curve()
    shift = S44.compute_rocky_threshold_shift(m_ref, r_ref)
    print()
    nasa_all, rocky = S44.load_and_filter_nasa(m_ref, r_ref, shift)
    rocky_win = S44.restrict_to_window(rocky)
    print(f"Rocky NASA planets in window: {len(rocky_win):,}")
    color_map, major, counts = S44.build_facility_styles(rocky_win)
    print()

    # Flat control catalogue + detectors.
    catalog = get_or_build_catalog(cache_csv, rebuild=args.rebuild,
                                   n_planets=args.n_planets, seed=args.seed)
    print(f"--> Running Kepler / TESS detectors on {len(catalog):,} flat planets...")
    kepler = prepare_flat(run_kepler(catalog), "Kepler")
    tess   = prepare_flat(run_tess(catalog),   "TESS")

    plot_combined_flat(kepler, tess, rocky_win, color_map, major, counts)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""Makes 3x3 plot showing how well telescopes can detect certain planets (fig 3 in the paper)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullLocator

from tools.paths import REPO_ROOT, PAPER_FIGURES_DIR, _EXOPLANET_CSV_DIR
ROOT = Path(REPO_ROOT)

from science.telescopes.detection import (
    build_stellar_selection_panel,
    missed_fraction_in_window,
)
from science.catalogs import restrict_science_window, load_and_filter_nasa
from science.statistics import binned_fraction_2d, fit_quantile_power_law
from science.physics import load_rocky_reference_curve, compute_rocky_threshold_shift
from plotting.paper_figures import rocky_scatter_gaia60pc as rocky_scatter

PAPER_FIG_DIR = Path(PAPER_FIGURES_DIR)
NASA_FLAGS_CACHE = _EXOPLANET_CSV_DIR / "pscomppars_transiting_mass_insol.csv"

FIGURE_STYLE = {
    "font.size": 28, "axes.titlesize": 36, "axes.labelsize": 28,
    "legend.fontsize": 28, "xtick.labelsize": 26, "ytick.labelsize": 26,
}
BADGE_FONTSIZE = 26
PLANET_LABEL_FONTSIZE = BADGE_FONTSIZE - 1
COLORBAR_LABELSIZE = FIGURE_STYLE["legend.fontsize"]
COLORBAR_TICKSIZE = FIGURE_STYLE["legend.fontsize"]

# Which transit pipeline fills the transit row: "TESS" (paper Fig. 2) or
# "Kepler" (appendix comparison).
TRANSIT_MISSION = "TESS"

PERIOD_LIMS = (0.2, 20000.0)               # 0.2 d reaches the USP corner of the map
RADIUS_LIMS = (0.5, 2.2)
RV_MAG_TARGET = 12.0
MIN_BIN_COUNT = 2

# Per-column configuration. The M insolation range stops at 1e3 I_earth: an
# M dwarf at P >= 0.2 d cannot exceed ~1.5e3, so bins beyond that are
# unpopulatable at any sample size. 3 bins per decade in every column.
COLUMNS = {
    # G also gets a map-only supplementary draw: transits are rare at low
    # insolation around G stars, and the low-insolation large-radius region came out empty.
    "G": dict(teff=(5200.0, 6000.0), insol=(0.1, 1e4), n=3_000_000, seed=75,
              map_extra_n=6_000_000, map_extra_seed=175),
    "K": dict(teff=(3700.0, 5200.0), insol=(0.1, 1e4), n=2_500_000, seed=76),
    "M": dict(teff=(2300.0, 3700.0), insol=(0.1, 1e3), n=1_500_000, seed=77),
}
XBINS = {
    "G": np.logspace(-1, 4, 16),
    "K": np.logspace(-1, 4, 16),
    "M": np.logspace(-1, 3, 13),
}
# Linear radius axis from 0.6 R_earth, binned in 0.1 R_earth steps (an edge
# falls on the 1.4 R_earth desert boundary). The top bin spans 2.0-2.2: few
# planets above 2.1 R_earth are rocky, so a 2.1-2.2 bin would be mostly empty.
Y_LIMS = (0.6, rocky_scatter.RADIUS_LIMITS[1])
YBINS = np.append(np.round(np.arange(Y_LIMS[0], 2.0 + 1e-9, 0.1), 2), Y_LIMS[1])
Y_TICKS = [1.0, 1.5, 2.0]

ROWS = [
    ("Transit test", "transit_detected"),
    ("RV mass test", "rv_detected"),
    ("Transit + RV", "joint_detected"),
]


def build_column(stype: str, m_ref, r_ref) -> pd.DataFrame:
    cfg = COLUMNS[stype]
    print(f"  {stype}: drawing {cfg['n']:,} planets...", flush=True)
    panel, catalog, extra = build_stellar_selection_panel(
        cfg, m_ref, r_ref, radius_limits=RADIUS_LIMS, period_limits=PERIOD_LIMS,
        minimum_radius=rocky_scatter.RADIUS_LIMITS[0],
        transit_mission=TRANSIT_MISSION, rv_mag_target=RV_MAG_TARGET,
    )
    main = panel[panel["main"]]
    print(f"  {stype}: {cfg['n']:,} drawn -> {len(catalog):,} rocky transiting-geometry "
          f"candidates -> {int(main['transit_eligible'].sum()):,} transiting; "
          f"pass transit {int(main['transit_detected'].sum()):,}, "
          f"rv {int(main['rv_detected'].sum()):,}, "
          f"joint {int(main['joint_detected'].sum()):,}", flush=True)
    if extra is not None:
        print(f"     + {cfg['map_extra_n']:,} map-only draws -> {len(extra):,} candidates", flush=True)
    return panel


@plt.rc_context(FIGURE_STYLE)
def main():
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("flat_transit_rv_3x3.py — flat-Otegi selection maps (paper Fig. 2)")
    print(f"Transit pipeline: {TRANSIT_MISSION}")
    print("=" * 70)
    rows = list(ROWS)
    if TRANSIT_MISSION != "TESS":
        rows[0] = (f"Transit test ({TRANSIT_MISSION})", "transit_detected")
        rows[2] = (f"Transit ({TRANSIT_MISSION}) + RV", "joint_detected")
    m_ref, r_ref = load_rocky_reference_curve()
    shift = compute_rocky_threshold_shift(m_ref, r_ref)
    _, rocky = load_and_filter_nasa(NASA_FLAGS_CACHE, m_ref, r_ref, shift)
    rocky_win = restrict_science_window(
        rocky,
        insolation=rocky_scatter.INSOLATION_LIMITS,
        radius=rocky_scatter.RADIUS_LIMITS,
        stellar_types=rocky_scatter.STAR_ORDER,
    )
    # Kepler and K2 are the same spacecraft, so they share one color; everything
    # except TESS and Kepler/K2 is grouped as "Other observatories".
    rocky_win = rocky_win.assign(discovery_facility=rocky_win["discovery_facility"].replace(
        {"Kepler": "Kepler and K2", "K2": "Kepler and K2"}))
    color_map, major, counts = rocky_scatter.build_facility_styles(rocky_win, contrast_overrides=True)
    major = [f for f in major
             if f in ("Transiting Exoplanet Survey Satellite (TESS)", "Kepler and K2")]
    color_map["Kepler and K2"] = rocky_scatter.FACILITY_COLOR_OVERRIDES["K2"]  # brown reads on viridis
    lhs_mask = rocky_win["planet_label"].str.contains(r"LHS\s*1140\s*b", case=False,
                                                      na=False, regex=True)
    for st in COLUMNS:
        f_max = rocky_win.loc[rocky_win["stype_clean"] == st, "flux_p"].max()
        if np.isfinite(f_max) and f_max > COLUMNS[st]["insol"][1]:
            print(f"  [warn] {st}: hottest NASA rocky planet at {f_max:.0f} I_e exceeds "
                  f"the panel range {COLUMNS[st]['insol'][1]:g}")

    print("\nBuilding flat-Otegi columns:")
    panels = {st: build_column(st, m_ref, r_ref) for st in COLUMNS}

    fig, axes = plt.subplots(3, 3, figsize=(17, 15), sharex="col", sharey=True,
                             constrained_layout=True)
    mesh = None
    print("\nLow-insolation Super-Earth Desert detected fraction (S<50, R>1.4, among transiting):")
    for j, stype in enumerate(COLUMNS):
        panel = panels[stype]
        r = rocky_win[rocky_win["stype_clean"] == stype]
        xbins = XBINS[stype]
        fit = fit_quantile_power_law(r["flux_p"].values, r["radius_p"].values)
        for i, (row_name, test) in enumerate(rows):
            ax = axes[i, j]
            grid, _ = binned_fraction_2d(
                panel["insolation"], panel["radius"], panel[test],
                panel["transit_eligible"],
                xbins, YBINS, min_count=MIN_BIN_COUNT,
            )
            grid = grid.T
            mesh = ax.pcolormesh(xbins, YBINS, grid, shading="auto",
                                 vmin=0, vmax=1, cmap=rocky_scatter.CMAP_DETECTED)
            rocky_scatter._overlay_rocky_by_facility(ax, r[~lhs_mask.loc[r.index]], color_map, major,
                                           star_candidates=False)
            for _, row in r[lhs_mask.loc[r.index]].iterrows():
                c = color_map.get(row["discovery_facility"], rocky_scatter.OTHER_COLOR)
                ax.errorbar([row["flux_p"]], [row["radius_p"]],
                            yerr=[[abs(row["radius_err_minus"])], [abs(row["radius_err_plus"])]],
                            fmt="o", ms=14, color=c, mec="white", mew=1.8,
                            elinewidth=1.5, capsize=4, ecolor=c, zorder=7)
                ax.annotate("LHS 1140 b", xy=(row["flux_p"], row["radius_p"]),
                            xytext=(14, -6), textcoords="offset points", va="center",
                            fontsize=PLANET_LABEL_FONTSIZE, color="black", zorder=8,
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                                      alpha=0.8))
            # Solar-system reference points on the G column: Earth and Venus
            # (I = 1.91 I_earth, R = 0.949 R_earth).
            if stype == "G":
                for name, flux, radius, offset, ha in [("Earth", 1.0, 1.0, (0, 16), "center"),
                                                       ("Venus", 1.91, 0.949, (12, -18), "left")]:
                    ax.plot([flux], [radius], "o", ms=11, mfc="white", mec="black",
                            mew=1.8, zorder=7)
                    ax.annotate(name, xy=(flux, radius), xytext=offset,
                                textcoords="offset points", ha=ha, va="center",
                                fontsize=PLANET_LABEL_FONTSIZE, color="white", zorder=8)

            xlo = xbins[0]
            ax.fill_between([xlo, rocky_scatter.LOW_INSOLATION_REGION_INSOL], rocky_scatter.LOW_INSOLATION_REGION_RADIUS,
                            Y_LIMS[1], color="red", alpha=0.15, zorder=1.5, lw=0)
            # Bold Low-insolation Super-Earth Desert outline: bottom edge (R=1.4) + right edge (S=50).
            ax.plot([xlo, rocky_scatter.LOW_INSOLATION_REGION_INSOL], [rocky_scatter.LOW_INSOLATION_REGION_RADIUS] * 2,
                    color="red", lw=2.6, zorder=6)
            ax.plot([rocky_scatter.LOW_INSOLATION_REGION_INSOL, rocky_scatter.LOW_INSOLATION_REGION_INSOL],
                    [rocky_scatter.LOW_INSOLATION_REGION_RADIUS, Y_LIMS[1]], color="red", lw=2.6, zorder=6)
            fn, n_denom, n_missed = missed_fraction_in_window(
                panel, test, max_insolation=rocky_scatter.LOW_INSOLATION_REGION_INSOL,
                min_radius=rocky_scatter.LOW_INSOLATION_REGION_RADIUS,
            )
            if fn is not None:
                n_pass = n_denom - n_missed
                ax.text(0.03, 0.97, f"{1 - fn:.0%} detected",
                        transform=ax.transAxes, va="top", ha="left", fontsize=BADGE_FONTSIZE,
                        color="darkred",
                        bbox=dict(boxstyle="round", fc="white", ec="darkred", alpha=0.85),
                        zorder=8)
                print(f"  {row_name:14s} {stype}: {1 - fn:.1%} detected  ({n_pass}/{n_denom})")

            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_yticks(Y_TICKS)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_xlim(xbins[0], xbins[-1]); ax.set_ylim(Y_LIMS)
            # Inward ticks, so tick marks do not open gaps between touching panels.
            ax.tick_params(which="both", direction="in", top=True, right=True)
            # Columns sit edge to edge: only the first keeps radius labels, the
            # inner columns drop their 10^-1 label (it would land on the
            # neighbouring column's last tick), and each column's last label is
            # right-aligned so it does not overhang into the next column.
            ticks = 10.0 ** np.arange(round(np.log10(xbins[0])), round(np.log10(xbins[-1])) + 1)
            labels = [f"$10^{{{int(round(np.log10(t)))}}}$" for t in ticks]
            if j > 0:
                ax.tick_params(axis="y", labelleft=False)
                labels[0] = ""
            ax.set_xticks(ticks, labels)
            tick_labels = ax.get_xticklabels()
            if tick_labels:  # upper rows share x and draw no tick labels
                tick_labels[-1].set_horizontalalignment("right")
            if i == 0:
                ax.set_title(f"{stype} stars")
            if i == 2:
                ax.set_xlabel(r"Insolation flux [$I_\oplus$]")
            if j == 0:
                ax.set_ylabel(f"{row_name}\nPlanet radius [$R_\\oplus$]")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[f], markersize=10,
               label=rocky_scatter.FACILITY_RELABEL.get(f, f))
        for f in major
    ]
    if len(rocky_win) > sum(counts[f] for f in major):
        handles.append(Line2D([0], [0], marker="o", linestyle="", color=rocky_scatter.OTHER_COLOR,
                              markersize=10, label="Other observatories"))
    handles += [
        Patch(facecolor="red", alpha=0.15, edgecolor="red", lw=2.0,
              label=(r"$R>%.1f\,R_\oplus$, $I<%g\,I_\oplus$"
                     % (rocky_scatter.LOW_INSOLATION_REGION_RADIUS, rocky_scatter.LOW_INSOLATION_REGION_INSOL))),
    ]
    # Bottom annotations are manually anchored so the horizontal colorbar sits
    # on the left while the legend stacks to its right.
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.01, -0.02),
               ncol=1, framealpha=0.9)
    cbar_ax = fig.add_axes([0.095, -0.085, 0.54, 0.035])
    cbar = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal",
                        format=FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
    cbar.ax.tick_params(labelsize=COLORBAR_TICKSIZE)
    cbar.set_label("Percent planets detected", fontsize=COLORBAR_LABELSIZE)
    # Panels sit edge to edge with no gaps between rows or columns.
    fig.get_layout_engine().set(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)

    suffix = "" if TRANSIT_MISSION == "TESS" else f"_{TRANSIT_MISSION.lower()}"
    out = PAPER_FIG_DIR / f"flat_transit_rv_3x3_otegi{suffix}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

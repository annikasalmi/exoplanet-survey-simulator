"""Paper selection map (flat_transit_rv_3x3_otegi.png): rows = TESS transit, RV mass (best of
HARPS/NIRPS), both; columns = G, K, M hosts. Background = rocky planets from
the flat Otegi baseline.
Run: python plotting/scripts/analysis/flat_transit_rv_3x3.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from tools.paths import REPO_ROOT, ANALYSIS_DIR, PAPER_FIGURES_DIR
ROOT = Path(REPO_ROOT)

from science.populations.universes.flat_baseline import DEFAULTS, flat_baseline
from science.telescopes.kepler.detection_model import KeplerData
from science.telescopes.tess.detection_model import TESSData
from science.telescopes.rv.detection_model import RVData
from plotting.scripts.analysis import rocky_scatter_gaia60pc as rocky_scatter

OUT_DIR = Path(ANALYSIS_DIR) / "flat_transit_rv_3x3"
PAPER_FIG_DIR = Path(PAPER_FIGURES_DIR)

FIGURE_STYLE = {
    "font.size": 21, "axes.titlesize": 26, "axes.labelsize": 23,
    "legend.fontsize": 21, "xtick.labelsize": 19, "ytick.labelsize": 19,
}

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
    # insolation around G stars, and the cold large-radius corner came out empty.
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

ROWS = [("Transit test", "transit"), ("RV mass test", "rv"), ("Transit + RV", "joint")]


def detect(cat: pd.DataFrame) -> pd.DataFrame:
    """Run the transit and RV tests on one batch of transiting-geometry planets."""
    if TRANSIT_MISSION == "Kepler":
        kep = KeplerData(cat.copy(), source="ppop").determine_detectable()
        denom = kep["transiting_geometric"].astype(bool).to_numpy()
        det_transit = kep["detected"].astype(bool).to_numpy() & denom
    else:
        tess = TESSData(cat.copy(), source="ppop", use_cdpp_tables=False).determine_detectable()
        denom = (tess["tess_observed"].astype(bool)
                 & tess["tess_transiting_geometric"].astype(bool)).to_numpy()
        det_transit = tess["detected"].astype(bool).to_numpy() & denom
    # Paper operator: mass measured if EITHER channel reaches K/sigma_K >= 5 on a
    # host with band-mag <= 12 in that channel's band.
    det_rv = np.zeros(len(cat), bool)
    for inst in ("HARPS", "NIRPS"):
        out = RVData(cat.copy(), source="ppop", instrument=inst).determine_detectable()
        mag = pd.to_numeric(out["rv_mag"], errors="coerce").to_numpy()
        det_rv |= out["rv_detected"].astype(bool).to_numpy() & (mag <= RV_MAG_TARGET)

    return pd.DataFrame({
        "flux_p": cat["flux_p"].to_numpy(),
        "radius_p": cat["radius_p"].to_numpy(),
        "denom": denom,
        "transit": det_transit,
        "rv": det_rv & denom,
        "joint": det_transit & det_rv,
    })


def rocky_transiting(cat: pd.DataFrame, m_ref, r_ref, r_min: float) -> pd.DataFrame:
    """Keep rocky planets above r_min that pass the transit-geometry prefilter."""
    thr = rocky_scatter.rocky_threshold_at_mass(cat["mass_p"].to_numpy(), m_ref, r_ref, 0.0)
    rocky = (cat["radius_p"].to_numpy() <= thr) & np.isfinite(thr)
    cat = cat[rocky & (cat["radius_p"].to_numpy() > r_min)].copy()
    # Geometric transit prefilter with the detector's own criterion, so the
    # per-row TESS SNR loop only ever sees transiting planets.
    rs_au = cat["radius_s"].to_numpy() * TESSData.R_SUN_AU
    rp_au = cat["radius_p"].to_numpy() * TESSData.R_EARTH_AU
    b = cat["semimajor_p"].to_numpy() * np.abs(np.cos(cat["inc_p"].to_numpy())) / rs_au
    return cat[b <= 1.0 + rp_au / rs_au].copy().reset_index(drop=True)


def build_column(stype: str, m_ref, r_ref) -> pd.DataFrame:
    cfg = COLUMNS[stype]
    cat = flat_baseline(
        cfg["n"], seed=cfg["seed"],
        radius_lims=RADIUS_LIMS, mass_lims=DEFAULTS["mass_lims"], period_lims=PERIOD_LIMS,
        teff_lims=cfg["teff"], insol_lims=cfg["insol"],
    )
    # Catalogue floor stays at the paper's 0.6 R_earth: the detectors draw random
    # numbers per planet, so changing the catalogue would shift the numbers.
    cat = rocky_transiting(cat, m_ref, r_ref, rocky_scatter.RADIUS_LIMITS[0])
    panel = detect(cat).assign(main=True)
    denom = panel["denom"].to_numpy()
    print(f"  {stype}: {cfg['n']:,} drawn -> {len(cat):,} rocky transiting-geometry "
          f"candidates -> {int(denom.sum()):,} transiting; "
          f"pass transit {int(panel['transit'].sum()):,}, rv {int(panel['rv'].sum()):,}, "
          f"joint {int(panel['joint'].sum()):,}")
    if cfg.get("map_extra_n"):
        # Map-only planets (main=False): they fill sparse bins of the background
        # map but are left out of the desert percentages, which stay as published.
        extra = flat_baseline(
            cfg["map_extra_n"], seed=cfg["map_extra_seed"],
            radius_lims=RADIUS_LIMS, mass_lims=DEFAULTS["mass_lims"], period_lims=PERIOD_LIMS,
            teff_lims=cfg["teff"], insol_lims=cfg["insol"],
        )
        extra = rocky_transiting(extra, m_ref, r_ref, rocky_scatter.RADIUS_LIMITS[0])
        panel = pd.concat([panel, detect(extra).assign(main=False)], ignore_index=True)
        print(f"     + {cfg['map_extra_n']:,} map-only draws -> {len(extra):,} candidates")
    return panel


def fraction_grid(panel: pd.DataFrame, test: str, xbins: np.ndarray):
    x, y = panel["flux_p"].to_numpy(), panel["radius_p"].to_numpy()
    d, n = panel["denom"].to_numpy(), panel[test].to_numpy()
    total, _, _ = np.histogram2d(x[d], y[d], bins=[xbins, YBINS])
    num, _, _ = np.histogram2d(x[n], y[n], bins=[xbins, YBINS])
    frac = np.divide(num, total, out=np.full_like(num, np.nan), where=total > 0)
    frac[total < MIN_BIN_COUNT] = np.nan
    return frac.T, total.T


def window_stats(panel: pd.DataFrame, test: str):
    """Detected among transiting rocky planets in the Cold Super-Earth Desert (S<50, R>1.4)."""
    panel = panel[panel["main"]]  # map-only supplementary draws are excluded
    region = ((panel["flux_p"] < rocky_scatter.COLD_CORNER_INSOL)
              & (panel["radius_p"] > rocky_scatter.COLD_CORNER_RADIUS)
              & panel["denom"]).to_numpy()
    n = int(region.sum())
    if n == 0:
        return None, 0, 0
    n_pass = int((region & panel[test].to_numpy()).sum())
    return (n - n_pass) / n, n, n - n_pass


def main(paper_copy: bool = True):
    with plt.rc_context(FIGURE_STYLE):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print("=" * 70)
        print("flat_transit_rv_3x3.py — flat-Otegi selection maps (paper Fig. 2)")
        print(f"Transit pipeline: {TRANSIT_MISSION}")
        print("=" * 70)
        rows = list(ROWS)
        if TRANSIT_MISSION != "TESS":
            rows[0] = (f"Transit test ({TRANSIT_MISSION})", "transit")
            rows[2] = (f"Transit ({TRANSIT_MISSION}) + RV", "joint")
        m_ref, r_ref = rocky_scatter.load_rocky_reference_curve()
        shift = rocky_scatter.compute_rocky_threshold_shift(m_ref, r_ref)
        _, rocky = rocky_scatter.load_and_filter_nasa(m_ref, r_ref, shift)
        rocky_win = rocky_scatter.restrict_to_window(rocky)
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
        print("\nCold Super-Earth Desert detected fraction (S<50, R>1.4, among transiting):")
        for j, stype in enumerate(COLUMNS):
            panel = panels[stype]
            r = rocky_win[rocky_win["stype_clean"] == stype]
            xbins = XBINS[stype]
            fit = rocky_scatter.fit_90pct_line(r["flux_p"].values, r["radius_p"].values)
            for i, (row_name, test) in enumerate(rows):
                ax = axes[i, j]
                grid, _ = fraction_grid(panel, test, xbins)
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
                                fontsize=20, color="black", zorder=8,
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
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
                                    fontsize=20, color="white", zorder=8)

                xlo = xbins[0]
                ax.fill_between([xlo, rocky_scatter.COLD_CORNER_INSOL], rocky_scatter.COLD_CORNER_RADIUS,
                                Y_LIMS[1], color="red", alpha=0.15, zorder=1.5, lw=0)
                # Bold Cold Super-Earth Desert outline: bottom edge (R=1.4) + right edge (S=50).
                ax.plot([xlo, rocky_scatter.COLD_CORNER_INSOL], [rocky_scatter.COLD_CORNER_RADIUS] * 2,
                        color="red", lw=2.6, zorder=6)
                ax.plot([rocky_scatter.COLD_CORNER_INSOL, rocky_scatter.COLD_CORNER_INSOL],
                        [rocky_scatter.COLD_CORNER_RADIUS, Y_LIMS[1]], color="red", lw=2.6, zorder=6)
                fn, n_denom, n_missed = window_stats(panel, test)
                if fn is not None:
                    n_pass = n_denom - n_missed
                    ax.text(0.03, 0.97, f"{1 - fn:.0%} detected",
                            transform=ax.transAxes, va="top", ha="left", fontsize=19,
                            color="darkred",
                            bbox=dict(boxstyle="round", fc="white", ec="darkred", alpha=0.85),
                            zorder=8)
                    print(f"  {row_name:14s} {stype}: {1 - fn:.1%} detected  ({n_pass}/{n_denom})")

                ax.set_xscale("log")
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
                         % (rocky_scatter.COLD_CORNER_RADIUS, rocky_scatter.COLD_CORNER_INSOL))),
        ]
        # Anchored just below the figure (bbox_inches="tight" keeps it): with zero
        # layout padding, an "outside" legend would overlap the bottom x-labels.
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0),
                   ncol=len(handles), framealpha=0.9)
        # Horizontal colour bar under the panels (the legend sits below it), with
        # the pass fraction shown as a percentage.
        fig.colorbar(mesh, ax=axes.ravel().tolist(), location="bottom", shrink=0.6,
                     aspect=40, pad=0.02, label="Percent planets detected",
                     format=FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
        # Panels sit edge to edge with no gaps between rows or columns.
        fig.get_layout_engine().set(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)

        suffix = "" if TRANSIT_MISSION == "TESS" else f"_{TRANSIT_MISSION.lower()}"
        out = OUT_DIR / f"flat_transit_rv_3x3_otegi{suffix}.png"
        fig.savefig(out, dpi=170, bbox_inches="tight")
        print(f"\nSaved: {out}")
        if paper_copy:
            PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
            fig.savefig(PAPER_FIG_DIR / out.name, dpi=170, bbox_inches="tight")
            print(f"Saved paper copy: {PAPER_FIG_DIR / out.name}")
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

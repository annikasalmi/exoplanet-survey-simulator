"""
flat_transit_rv_3x3.py — the paper's selection-operator map (Figure 2).

3x3 grid on the insolation-radius plane:
    rows    = Transit test (TESS) | RV mass test (best of HARPS/NIRPS) | Transit + RV joint
    columns = G, K, M hosts (F dropped — too few NASA rocky planets, mostly blind)

Background = ROCKY flat-control planets (below the unshifted silicate curve,
silicon_curve.ddat). The flat universe draws radius/orbit/star flat, and masses
follow the Otegi et al. 2020 rocky relation R = 1.03 M^0.29 with 0.15 dex mass
scatter (the paper's default relation, per script 77). Pass fractions are among
TRANSITING rocky planets; both detectors run with their default calibrations
(Kepler x0.84 is unused here; TESS x0.66; RV needs none).

Sampling is stratified per spectral type with large N and per-column insolation
ranges (M capped at 1e3 I_earth, its physical short-period limit) so that no
plotted bin is left empty. Periods extend down to 0.2 d to reach the
ultra-short-period corner.

Run:
    python important_plots/flat_transit_rv_3x3.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import importlib.util
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

from run.flat_universe.uniform_generator import generate_flat_catalog
from run.ppop.flat_detect import run_kepler, run_tess, run_rv, TESSData


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S44 = _load("s44", ROOT / "scripts" / "statistical_analysis" / "rocky_scatter_gaia60pc.py")

OUT_DIR = ROOT / "my_outputs" / "flat_transit_rv_3x3"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR = ROOT / "paper" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
})

# Which transit pipeline fills the transit row: "TESS" (paper Fig. 2) or
# "Kepler" (appendix comparison). Patched by script 48's mapskep part.
TRANSIT_MISSION = "TESS"

OTEGI = dict(mr_C=1.03, mr_beta=0.29)      # R = 1.03 M^0.29, script 77's default relation
MR_SCATTER_DEX = 0.15
PERIOD_LIMS = (0.2, 20000.0)               # 0.2 d reaches the USP corner of the map
RADIUS_LIMS = (0.5, 2.2)
RV_MAG_TARGET = 12.0
MIN_BIN_COUNT = 2

# Per-column configuration. The M insolation range stops at 1e3 I_earth: an
# M dwarf at P >= 0.2 d cannot exceed ~1.5e3, so bins beyond that are
# unpopulatable at any sample size. 3 bins per decade in every column.
COLUMNS = {
    "G": dict(teff=(5200.0, 6000.0), insol=(0.1, 1e4), n=3_000_000, seed=75),
    "K": dict(teff=(3700.0, 5200.0), insol=(0.1, 1e4), n=2_500_000, seed=76),
    "M": dict(teff=(2300.0, 3700.0), insol=(0.1, 1e3), n=1_500_000, seed=77),
}
XBINS = {
    "G": np.logspace(-1, 4, 16),
    "K": np.logspace(-1, 4, 16),
    "M": np.logspace(-1, 3, 13),
}
YBINS = S44.PLANET_RADIUS_BINS
Y_LIMS = S44.RADIUS_LIMITS

ROWS = [("Transit test", "transit"), ("RV mass test", "rv"), ("Transit + RV", "joint")]


def build_column(stype: str, m_ref, r_ref) -> pd.DataFrame:
    cfg = COLUMNS[stype]
    cat = generate_flat_catalog(
        cfg["n"], seed=cfg["seed"],
        radius_lims=RADIUS_LIMS, period_lims=PERIOD_LIMS,
        teff_lims=cfg["teff"], insol_lims=cfg["insol"],
        mass_model="powerlaw", mass_scatter_dex=MR_SCATTER_DEX, **OTEGI,
    )
    thr = S44.rocky_threshold_at_mass(cat["mass_p"].to_numpy(), m_ref, r_ref, 0.0)
    rocky = (cat["radius_p"].to_numpy() <= thr) & np.isfinite(thr)
    cat = cat[rocky & (cat["radius_p"].to_numpy() > Y_LIMS[0])].copy()
    n_rocky = len(cat)

    # Geometric transit prefilter with the detector's own criterion, so the
    # per-row TESS SNR loop only ever sees transiting planets.
    rs_au = cat["radius_s"].to_numpy() * TESSData.R_SUN_AU
    rp_au = cat["radius_p"].to_numpy() * TESSData.R_EARTH_AU
    b = cat["semimajor_p"].to_numpy() * np.abs(np.cos(cat["inc_p"].to_numpy())) / rs_au
    cat = cat[b <= 1.0 + rp_au / rs_au].copy().reset_index(drop=True)

    if TRANSIT_MISSION == "Kepler":
        kep = run_kepler(cat)
        denom = kep["transiting_geometric"].astype(bool).to_numpy()
        det_transit = kep["detected"].astype(bool).to_numpy() & denom
    else:
        tess = run_tess(cat)
        denom = (tess["tess_observed"].astype(bool)
                 & tess["tess_transiting_geometric"].astype(bool)).to_numpy()
        det_transit = tess["detected"].astype(bool).to_numpy() & denom
    # Paper operator: mass measured if EITHER channel reaches K/sigma_K >= 5 on a
    # host with band-mag <= 12 in that channel's band (scripts 50/53 lineage).
    det_rv = np.zeros(len(cat), bool)
    for inst in ("HARPS", "NIRPS"):
        out = run_rv(cat, inst)
        mag = pd.to_numeric(out["rv_mag"], errors="coerce").to_numpy()
        det_rv |= out["rv_detected"].astype(bool).to_numpy() & (mag <= RV_MAG_TARGET)

    panel = pd.DataFrame({
        "flux_p": cat["flux_p"].to_numpy(),
        "radius_p": cat["radius_p"].to_numpy(),
        "denom": denom,
        "transit": det_transit,
        "rv": det_rv & denom,
        "joint": det_transit & det_rv,
    })
    print(f"  {stype}: {cfg['n']:,} drawn -> {n_rocky:,} rocky in window -> "
          f"{len(panel):,} transit-geometry candidates -> {int(denom.sum()):,} transiting; "
          f"pass transit {int(panel['transit'].sum()):,}, rv {int(panel['rv'].sum()):,}, "
          f"joint {int(panel['joint'].sum()):,}")
    return panel


def fraction_grid(panel: pd.DataFrame, test: str, xbins: np.ndarray):
    x, y = panel["flux_p"].to_numpy(), panel["radius_p"].to_numpy()
    d, n = panel["denom"].to_numpy(), panel[test].to_numpy()
    total, _, _ = np.histogram2d(x[d], y[d], bins=[xbins, YBINS])
    num, _, _ = np.histogram2d(x[n], y[n], bins=[xbins, YBINS])
    frac = np.divide(num, total, out=np.full_like(num, np.nan), where=total > 0)
    frac[total < MIN_BIN_COUNT] = np.nan
    return frac.T, total.T


def add_contours(ax, grid, xbins):
    xc = np.sqrt(xbins[:-1] * xbins[1:])
    yc = np.sqrt(YBINS[:-1] * YBINS[1:])
    try:
        cs = ax.contour(*np.meshgrid(xc, yc), grid, levels=[0.2, 0.5, 0.8],
                        colors="white", linewidths=0.8, alpha=0.8)
        ax.clabel(cs, fmt="%.1f", fontsize=8)
    except Exception:
        pass


def window_stats(panel: pd.DataFrame, test: str):
    """Detected among transiting rocky planets in the Cold Rocky Desert (S<50, R>1.4)."""
    region = ((panel["flux_p"] < S44.COLD_CORNER_INSOL)
              & (panel["radius_p"] > S44.COLD_CORNER_RADIUS)
              & panel["denom"]).to_numpy()
    n = int(region.sum())
    if n == 0:
        return None, 0, 0
    n_pass = int((region & panel[test].to_numpy()).sum())
    return (n - n_pass) / n, n, n - n_pass


def main():
    print("=" * 70)
    print("flat_transit_rv_3x3.py — flat-Otegi selection maps (paper Fig. 2)")
    print(f"Transit pipeline: {TRANSIT_MISSION}")
    print("=" * 70)
    rows = list(ROWS)
    if TRANSIT_MISSION != "TESS":
        rows[0] = (f"Transit test ({TRANSIT_MISSION})", "transit")
        rows[2] = (f"Transit ({TRANSIT_MISSION}) + RV", "joint")
    m_ref, r_ref = S44.load_rocky_reference_curve()
    shift = S44.compute_rocky_threshold_shift(m_ref, r_ref)
    _, rocky = S44.load_and_filter_nasa(m_ref, r_ref, shift)
    rocky_win = S44.restrict_to_window(rocky)
    color_map, major, counts = S44.build_facility_styles(rocky_win, contrast_overrides=True)
    for st in COLUMNS:
        f_max = rocky_win.loc[rocky_win["stype_clean"] == st, "flux_p"].max()
        if np.isfinite(f_max) and f_max > COLUMNS[st]["insol"][1]:
            print(f"  [warn] {st}: hottest NASA rocky planet at {f_max:.0f} I_e exceeds "
                  f"the panel range {COLUMNS[st]['insol'][1]:g}")

    print("\nBuilding flat-Otegi columns:")
    panels = {st: build_column(st, m_ref, r_ref) for st in COLUMNS}

    fig, axes = plt.subplots(3, 3, figsize=(19, 15), sharex="col", sharey=True,
                             constrained_layout=True)
    mesh = None
    print("\nCold Rocky Desert detected fraction (S<50, R>1.4, among transiting):")
    for j, stype in enumerate(COLUMNS):
        panel = panels[stype]
        r = rocky_win[rocky_win["stype_clean"] == stype]
        xbins = XBINS[stype]
        fit = S44.fit_90pct_line(r["flux_p"].values, r["radius_p"].values)
        for i, (row_name, test) in enumerate(rows):
            ax = axes[i, j]
            grid, _ = fraction_grid(panel, test, xbins)
            mesh = ax.pcolormesh(xbins, YBINS, grid, shading="auto",
                                 vmin=0, vmax=1, cmap=S44.CMAP_DETECTED)
            add_contours(ax, grid, xbins)
            S44._overlay_rocky_by_facility(ax, r, color_map, major, star_candidates=False)

            xlo = xbins[0]
            ax.fill_between([xlo, S44.COLD_CORNER_INSOL], S44.COLD_CORNER_RADIUS,
                            Y_LIMS[1], color="red", alpha=0.15, zorder=1.5, lw=0)
            # Bold Cold Rocky Desert outline: bottom edge (R=1.4) + right edge (S=50).
            ax.plot([xlo, S44.COLD_CORNER_INSOL], [S44.COLD_CORNER_RADIUS] * 2,
                    color="red", lw=2.6, zorder=6)
            ax.plot([S44.COLD_CORNER_INSOL, S44.COLD_CORNER_INSOL],
                    [S44.COLD_CORNER_RADIUS, Y_LIMS[1]], color="red", lw=2.6, zorder=6)
            fn, n_denom, n_missed = window_stats(panel, test)
            if fn is not None:
                n_pass = n_denom - n_missed
                ax.text(0.03, 0.97, f"Cold Rocky Desert: {1 - fn:.0%} detected\n({n_pass:,}/{n_denom:,})",
                        transform=ax.transAxes, va="top", ha="left", fontsize=12,
                        color="darkred",
                        bbox=dict(boxstyle="round", fc="white", ec="darkred", alpha=0.85),
                        zorder=8)
                print(f"  {row_name:14s} {stype}: {1 - fn:.1%} detected  ({n_pass}/{n_denom})")

            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(xbins[0], xbins[-1]); ax.set_ylim(Y_LIMS)
            S44._format_radius_axis(ax)
            ax.grid(True, which="both", alpha=0.18)
            if i == 0:
                ax.set_title(f"{stype} stars (N$_{{\\rm NASA}}$={len(r)})")
            if i == 2:
                ax.set_xlabel(r"Insolation flux [$I_\oplus$]")
            if j == 0:
                ax.set_ylabel(f"{row_name}\nPlanet radius [$R_\\oplus$]")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[f], markersize=7,
               label=f"{S44.short_facility(f)}  (N={counts[f]})")
        for f in major
    ]
    n_other = int(len(rocky_win) - sum(counts[f] for f in major))
    if n_other > 0:
        handles.append(Line2D([0], [0], marker="o", linestyle="", color=S44.OTHER_COLOR,
                              markersize=7, label=f"Other facilities  (N={n_other})"))
    handles += [
        Patch(facecolor="red", alpha=0.15, edgecolor="red", lw=2.0,
              label=(r"Cold Rocky Desert ($S<50\,S_\oplus$, $R>%.2f\,R_\oplus$)"
                     % S44.COLD_CORNER_RADIUS)),
    ]
    ncol = (len(handles) + 1) // 2  # widen the legend so it fits in exactly 2 rows
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol, fontsize=12,
               framealpha=0.9,
               title="Observed rocky planets, colored by discovery facility",
               title_fontsize=13)
    fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.55, pad=0.01,
                 label="Pass fraction (transiting rocky planets)")

    suffix = "" if TRANSIT_MISSION == "TESS" else f"_{TRANSIT_MISSION.lower()}"
    out = OUT_DIR / f"flat_transit_rv_3x3_otegi{suffix}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    fig.savefig(PAPER_FIG_DIR / out.name, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")
    print(f"Saved paper copy: {PAPER_FIG_DIR / out.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
DOWNLOAD_NASA_DATA = False  # Set to True to download fresh data, False to use local CSV

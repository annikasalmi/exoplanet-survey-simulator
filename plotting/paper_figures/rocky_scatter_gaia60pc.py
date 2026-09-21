"""Rocky-planet figures: FGKM detection-fraction maps from stacked Gaia-60pc Kepler/TESS catalogs
with NASA rocky planets overlaid, plus the paper's rocky_mr_insolation_3panel / rocky_scatter_standalone.
Run: python plotting/scripts/analysis/rocky_scatter_gaia60pc.py [--full]
The paper figures need no universes. --full adds the maps, from the 10 Gaia-60pc
universes the Kepler/TESS lines in `sim.py` write (~3-4 h).
"""

from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from tools.paths import (
    SILICON_CURVE, ANALYSIS_DIR, PAPER_FIGURES_DIR, KEPLER_DATA_DIR,
    TESS_DATA_DIR, REPO_ROOT, _EXOPLANET_CSV_DIR,
)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, NullFormatter

from science.catalogs import load_and_filter_nasa, restrict_science_window
from science.physics import (
    compute_rocky_threshold_shift,
    load_mass_radius_curve,
    load_rocky_reference_curve,
    radius_on_curve,
)
from science.statistics import (
    binned_fraction_2d,
    fit_quantile_power_law,
)
from science.telescopes.detection import prepare_kepler_catalog, prepare_tess_catalog


ROOT = REPO_ROOT
N_UNIVERSES = 10

KEPLER_PPOP_DIR = KEPLER_DATA_DIR / "Gaia"
TESS_PPOP_DIR   = TESS_DATA_DIR / "Gaia"
REF_CURVE_PATH = REPO_ROOT / "science" / "telescopes" / "kepler" / "reference_curves" / "ref.ddat"
ROCKY_CURVE_PATH = SILICON_CURVE
ROCKY_CURVE_LABEL = "silicate rocky curve"
NASA_FLAGS_CACHE = _EXOPLANET_CSV_DIR / "pscomppars_transiting_mass_insol.csv"
OUT_DIR = ANALYSIS_DIR / "rocky_scatter_gaia60pc"
PAPER_FIG_DIR = PAPER_FIGURES_DIR


def _ppop_files(directory: Path, stem: str) -> list[Path]:
    """Return <stem>_<i>.csv for i in 0..N_UNIVERSES-1, in order. Raises if none exist."""
    wanted = [directory / f"{stem}_{i}.csv" for i in range(N_UNIVERSES)]
    present = [f for f in wanted if f.exists()]
    if not present:
        raise FileNotFoundError(
            f"No P-Pop catalogs found in {directory} for stem '{stem}' "
            f"(expected {stem}_0.csv .. {stem}_{N_UNIVERSES - 1}.csv); "
            "run the Kepler/TESS lines in `sim.py` first (~3-4 h)"
        )
    missing = [f.name for f in wanted if not f.exists()]
    if missing:
        print(f"  [warn] {len(missing)} expected universe(s) missing, "
              f"stacking {len(present)}: missing {missing}")
    return present

# ── LHS 1140 b anchor ────────────────────────────────────────────────────────

LHS1140B_MASS_MEARTH   = 5.60
LHS1140B_RADIUS_REARTH = 1.730

# ── NASA quality settings ────────────────────────────────────────────────────

FORCE_REDOWNLOAD_NASA    = False
DOWNLOAD_NASA_IF_MISSING = False

EXCLUDE_MASS_LIMITS       = True
EXCLUDE_RADIUS_LIMITS     = True
REQUIRE_TWO_SIDED_MASS    = True
REQUIRE_TWO_SIDED_RADIUS  = True

MAX_MASS_REL_UNCERTAINTY   = 0.25
MAX_RADIUS_REL_UNCERTAINTY = 0.08

# ── Grid / detection settings ─────────────────────────────────────────────────

STAR_ORDER = ["F", "G", "K", "M"]

# Baseline detection case for both missions.
KEPLER_MES_THRESHOLD = 7.1
KEPLER_CDPP_SCALE    = 1.0
TESS_SNR_THRESHOLD   = 7.1
TESS_NOISE_SCALE     = 1.0
CASE_NAME            = "baseline"

RADIUS_LIMITS     = (0.6, 2.2)
INSOLATION_LIMITS = (0.1, 1e4)

INSOLATION_BINS    = np.logspace(np.log10(INSOLATION_LIMITS[0]), np.log10(INSOLATION_LIMITS[1]), 16)
PLANET_RADIUS_BINS = np.logspace(np.log10(RADIUS_LIMITS[0]),     np.log10(RADIUS_LIMITS[1]), 12)

MIN_BIN_COUNT = 2
CMAP_DETECTED = "viridis"

# Facilities with at least this many rocky planets in-window get their own color.
FACILITY_MIN_COUNT = 2   # "> 2" planets  →  strictly greater than 2

# Pretty short names for legend labels.
FACILITY_RELABEL = {
    "Transiting Exoplanet Survey Satellite (TESS)": "TESS",
    "Next-Generation Transit Survey (NGTS)": "NGTS",
}







# ── Grid helpers ──────────────────────────────────────────────────────────────

def _add_contours(ax, grid: np.ndarray):
    if not np.isfinite(grid).any():
        return
    X, Y = np.meshgrid(
        np.sqrt(INSOLATION_BINS[:-1] * INSOLATION_BINS[1:]),
        np.sqrt(PLANET_RADIUS_BINS[:-1] * PLANET_RADIUS_BINS[1:]),
    )
    try:
        cs = ax.contour(X, Y, grid, levels=[0.2, 0.5, 0.8],
                        colors="white", linewidths=0.8, alpha=0.8)
        ax.clabel(cs, fmt="%.1f", fontsize=7)
    except Exception:
        pass


# Plain radius tick positions/labels (no 4x10^0 style scientific notation).
RADIUS_TICKS = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]


def _format_radius_axis(ax):
    """Label the log radius axis with plain numbers (0.6, 1, 2), even on shared-y panels."""
    ticks = [t for t in RADIUS_TICKS if RADIUS_LIMITS[0] <= t <= RADIUS_LIMITS[1]]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", labelleft=True)


def _setup_axis(ax, title: str, show_xlabel: bool = True, show_ylabel: bool = True):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(INSOLATION_LIMITS)
    ax.set_ylim(RADIUS_LIMITS)
    ax.set_title(title, fontsize=10)
    if show_xlabel:
        ax.set_xlabel(r"Insolation flux [$I_\oplus$]")
    if show_ylabel:
        ax.set_ylabel(r"Planet radius [$R_\oplus$]")
    _format_radius_axis(ax)
    ax.grid(True, which="both", alpha=0.18)


def draw_90pct_line(ax, flux, radius, color="black", lw=2.0, label=None,
                    x_extent=None):
    """Fit and draw the 90% line; returns (slope, intercept) or None.
    x_extent=(lo, hi) sets the flux range drawn (default: the data's range).
    """
    fit = fit_quantile_power_law(flux, radius, quantile=0.90)
    if fit is None:
        return None
    slope, intercept = fit
    if x_extent is None:
        f = np.asarray(flux, dtype=float)
        f = f[np.isfinite(f) & (f > 0)]
        if f.size == 0:
            return None
        lo, hi = f.min(), f.max()
    else:
        lo, hi = x_extent
    x = np.logspace(np.log10(lo), np.log10(hi), 200)
    ax.plot(x, 10 ** (slope * np.log10(x) + intercept), color=color, lw=lw, ls="--",
            alpha=0.9, zorder=5, label=label)
    return slope, intercept


COLD_INSOLATION = 10.0  # "I < 10" low-insolation boundary


def false_negative_prob(ppop_panel: pd.DataFrame, slope: float, intercept: float):
    """P-Pop false-negative probability for I < 10 and radius above the 90% line: the fraction of
    transiting P-Pop planets there the survey misses. Returns (fn_prob, n_denominator, n_missed).
    """
    f = pd.to_numeric(ppop_panel["flux_p"], errors="coerce")
    rr = pd.to_numeric(ppop_panel["radius_p"], errors="coerce")
    in_region = (
        (f < COLD_INSOLATION)
        & (rr > 10 ** (slope * np.log10(f) + intercept))
        & ppop_panel["denominator_case"].to_numpy()
    )
    n_denom = int(in_region.sum())
    if n_denom == 0:
        return None, 0, 0
    n_det = int((in_region & ppop_panel["detected_case"].to_numpy()).sum())
    n_missed = n_denom - n_det
    return n_missed / n_denom, n_denom, n_missed


# ── Facility colors ───────────────────────────────────────────────────────────

FACILITY_PALETTE = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Set2").colors)

# Fixed colors for facilities whose default color blends into the viridis background (TESS blue,
# K2 green): magenta and brown stay distinct. Used only by figures that opt in.
FACILITY_COLOR_OVERRIDES = {
    "TESS": "#ff00ff",
    "K2": "#8b4513",
}


def build_facility_styles(rocky_win: pd.DataFrame, contrast_overrides: bool = False):
    """Color each discovery facility with more than FACILITY_MIN_COUNT rocky planets in the window.
    Returns (color_map, major_facilities_in_order, counts_series).
    """
    counts = rocky_win["discovery_facility"].fillna("Unknown").value_counts()
    major = [f for f, c in counts.items() if c > FACILITY_MIN_COUNT]
    color_map = {f: FACILITY_PALETTE[i % len(FACILITY_PALETTE)] for i, f in enumerate(major)}
    if contrast_overrides:
        for f in major:
            override = FACILITY_COLOR_OVERRIDES.get(FACILITY_RELABEL.get(f, f))
            if override:
                color_map[f] = override
    return color_map, major, counts


OTHER_COLOR = "0.55"

# The low-insolation super-Earth desert candidates (relaxed cuts; paper_v2 Table 1), all starred
# equally in overlay panels — LHS 1140 b is not singled out.
DESERT_CANDIDATE_PATTERNS = [
    r"LHS\s*1140\s*b", r"TOI-?1452\s*b", r"LHS\s*1903\s*e",
    r"TOI-?198\s*b", r"TOI-?771\s*b", r"TOI-?1468\s*b",
]
DESERT_CANDIDATE_REGEX = "|".join(f"(?:{p})" for p in DESERT_CANDIDATE_PATTERNS)


def _overlay_rocky_by_facility(ax, sub: pd.DataFrame, color_map: dict, major: list[str],
                               star_candidates: bool = True):
    """Overlay rocky planets in one panel, colored & error-barred by facility."""
    if len(sub) == 0:
        return

    yerr_hi = sub["radius_err_plus"].abs().fillna(0).values
    yerr_lo = sub["radius_err_minus"].abs().fillna(0).values
    fac = sub["discovery_facility"].fillna("Unknown").values

    # Major facilities (own color), then everything else as "Other".
    for f in major:
        m = fac == f
        if not m.any():
            continue
        ax.errorbar(
            sub["flux_p"].values[m], sub["radius_p"].values[m],
            yerr=[yerr_lo[m], yerr_hi[m]],
            fmt="o", ms=4, color=color_map[f], alpha=0.85,
            elinewidth=0.8, capsize=2, ecolor=color_map[f],
            zorder=4,
        )

    other = ~np.isin(fac, major)
    if other.any():
        ax.errorbar(
            sub["flux_p"].values[other], sub["radius_p"].values[other],
            yerr=[yerr_lo[other], yerr_hi[other]],
            fmt="o", ms=4, color=OTHER_COLOR, alpha=0.70,
            elinewidth=0.7, capsize=1.5, ecolor=OTHER_COLOR,
            zorder=3,
        )

    # Star every desert candidate that landed in this panel, all styled alike.
    if not star_candidates:
        return
    cand_mask = sub["planet_label"].str.contains(
        DESERT_CANDIDATE_REGEX, case=False, na=False, regex=True)
    for _, row in sub[cand_mask].iterrows():
        ax.scatter(
            [row["flux_p"]], [row["radius_p"]],
            s=70, marker="*", color="gold", edgecolors="darkred",
            linewidths=0.9, zorder=6,
        )


# ── Combined 2x4 figure ───────────────────────────────────────────────────────

def plot_combined(kepler: pd.DataFrame, tess: pd.DataFrame,
                  rocky_win: pd.DataFrame, shift: float,
                  color_map: dict, major: list[str], counts) -> Path:
    print("\nFacilities with > {} rocky planets in window:".format(FACILITY_MIN_COUNT))
    for f in major:
        print(f"  {FACILITY_RELABEL.get(f, f):<10s} N={counts[f]}")
    n_other = int(len(rocky_win) - sum(counts[f] for f in major))
    print(f"  Other      N={n_other}")

    rows = [
        ("Kepler", kepler),
        ("TESS",   tess),
    ]

    fig, axes = plt.subplots(
        2, 4, figsize=(24, 11), sharex=True, sharey=True, constrained_layout=True
    )
    mesh = None

    print("\nLow-insolation-region false-negative probabilities (I<10, radius above 90% fit):")
    for i, (mission, ppop) in enumerate(rows):
        for j, stype in enumerate(STAR_ORDER):
            ax = axes[i, j]
            p = ppop[ppop["stype_clean"] == stype].copy()
            r = rocky_win[rocky_win["stype_clean"] == stype].copy()

            n_bg = int(p["denominator_case"].sum())  # P-Pop planets forming the background
            det_grid, _ = binned_fraction_2d(
                p["flux_p"],
                p["radius_p"],
                p["detected_case"],
                p["denominator_case"],
                INSOLATION_BINS,
                PLANET_RADIUS_BINS,
                min_count=MIN_BIN_COUNT,
            )
            det_grid = det_grid.T
            if np.isfinite(det_grid).any():
                mesh = ax.pcolormesh(
                    INSOLATION_BINS, PLANET_RADIUS_BINS, det_grid,
                    shading="auto", vmin=0, vmax=1, cmap=CMAP_DETECTED,
                )
                _add_contours(ax, det_grid)
            else:
                ax.text(0.5, 0.5, "no P-Pop data", transform=ax.transAxes,
                        ha="center", va="center", color="0.5")

            _overlay_rocky_by_facility(ax, r, color_map, major)

            # 90% upper-bound fit + low-insolation false-negative region (G/K/M panels).
            fn_note = ""
            if stype in ("G", "K", "M"):
                fit = draw_90pct_line(
                    ax, r["flux_p"].values, r["radius_p"].values,
                    color="black", lw=2.0, x_extent=INSOLATION_LIMITS,
                )
                if fit is not None:
                    slope, intercept = fit
                    # Shade the low-insolation (I<10), above-the-line region.
                    xreg = np.logspace(np.log10(INSOLATION_LIMITS[0]),
                                       np.log10(COLD_INSOLATION), 100)
                    ax.fill_between(xreg, 10 ** (slope * np.log10(xreg) + intercept),
                                    RADIUS_LIMITS[1], color="red", alpha=0.12,
                                    zorder=1.5, lw=0)
                    fn, n_denom, n_missed = false_negative_prob(p, slope, intercept)
                    if fn is not None:
                        fn_note = (f"P(false neg | I<10, R>fit) = {fn:.0%}\n"
                                   f"({n_missed}/{n_denom} P-Pop missed)")
                        ax.text(0.03, 0.97, fn_note, transform=ax.transAxes,
                                va="top", ha="left", fontsize=7, color="darkred",
                                bbox=dict(boxstyle="round", fc="white",
                                          ec="darkred", alpha=0.85), zorder=8)
                        print(f"  {mission:6s} {stype}: "
                              f"P(false neg | I<10, R>fit) = {fn:.1%}  "
                              f"({n_missed}/{n_denom} P-Pop)")

            _setup_axis(
                ax,
                f"{mission} — {stype} stars\n"
                f"N_PPop(background) = {n_bg:,}   ·   N_rocky = {len(r)}",
                show_xlabel=(i == 1),
                show_ylabel=(j == 0),
            )

    # Shared facility legend.
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[f],
               markersize=7, label=f"{FACILITY_RELABEL.get(f, f)}  (N={counts[f]})")
        for f in major
    ]
    if n_other > 0:
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", color=OTHER_COLOR,
                   markersize=7, label=f"Other facilities  (N={n_other})")
        )
    handles.append(
        Line2D([0], [0], marker="*", linestyle="", color="gold",
               markeredgecolor="darkred", markersize=12, label="Low-insolation super-Earth desert candidates")
    )
    handles.append(
        Line2D([0], [0], color="black", lw=2.0, ls="--",
               label="90% upper-bound fit (≈90% of planets below; G/K/M)")
    )
    handles.append(
        Patch(facecolor="red", alpha=0.12,
              label="Low-insolation false-negative region (I<10, R>fit)")
    )
    # "outside" reserves space below the axes so the legend never overlaps the
    # bottom-row x-axis labels.
    fig.legend(
        handles=handles, loc="outside lower center",
        ncol=min(len(handles), 4), fontsize=9, framealpha=0.9,
        title="Rocky NASA planets — colored by discovery facility (telescope/mission)",
        title_fontsize=10,
    )

    fig.colorbar(mesh, ax=axes.ravel().tolist(),
                 label="Detected fraction", shrink=0.6, pad=0.01)

    fig.suptitle(
        "Kepler (top) & TESS (bottom) FGKM — P-Pop detected-fraction background "
        "+ rocky PSCompPars overlay  [baseline]\n"
        f"Rocky threshold = {ROCKY_CURVE_LABEL}, unshifted  |  "
        f"red error bars = two-sided radius uncertainty",
        fontsize=13,
    )

    out = OUT_DIR / "kepler_tess_2x4_rocky_fgkm_detection_baseline.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved combined 2x4 figure: {out}")
    return out


# ── Mass-radius diagnostic ───────────────────────────────────────────────────

STYPE_COLORS = {"F": "#e6ab02", "G": "#66a61e", "K": "#7570b3", "M": "#d95f02"}


def plot_mr_diagnostic(m_ref, r_ref, shift: float,
                       nasa_win: pd.DataFrame, rocky_win: pd.DataFrame,
                       color_map: dict, major: list[str], counts,
                       red_shift: float | None = None,
                       title: str | None = None,
                       out_name: str = "rocky_threshold_diagnostic_mass_radius.png") -> Path:
    """Mass-radius diagnostic for planets in the context window, colored by discovery
    facility with two-sided error bars.
    """
    XLIM = (0.0, 12.0)
    YLIM = (RADIUS_LIMITS[0] - 0.05, RADIUS_LIMITS[1])
    # red_shift lets callers draw the silicate curve unshifted (red_shift=0) or
    # with any other anchor offset; defaults to the LHS 1140 b shift.
    red_shift = shift if red_shift is None else red_shift

    m_line = np.linspace(XLIM[0] + 1e-3, XLIM[1], 600)
    # Red rocky threshold = silicate curve (m_ref, r_ref), anchored to LHS 1140 b.
    r_threshold = radius_on_curve(m_line, m_ref, r_ref, shift=red_shift)
    # Black dashed reference = pure-rock curve (ref.ddat), drawn only for comparison.
    if REF_CURVE_PATH.exists():
        _ref = np.loadtxt(REF_CURVE_PATH, comments="#")
        _mr, _rr = _ref[:, 0].astype(float), _ref[:, 1].astype(float)
        _ord = np.argsort(_mr)
        r_rocky = radius_on_curve(m_line, _mr[_ord], _rr[_ord])
    else:
        r_rocky = np.full_like(m_line, np.nan)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    ax.scatter(nasa_win["mass_p"], nasa_win["radius_p"],
               s=10, c="0.85", alpha=0.30, linewidths=0,
               label=f"PSCompPars quality-filtered, in-window (N={len(nasa_win):,})", zorder=1)

    # Rocky planets colored by discovery facility, thin cap-less error bars.
    fac = rocky_win["discovery_facility"].fillna("Unknown")
    for f in major:
        sub = rocky_win[fac == f]
        if len(sub) == 0:
            continue
        ax.errorbar(
            sub["mass_p"], sub["radius_p"],
            xerr=[sub["mass_err_minus"].abs().fillna(0).values,
                  sub["mass_err_plus"].abs().fillna(0).values],
            yerr=[sub["radius_err_minus"].abs().fillna(0).values,
                  sub["radius_err_plus"].abs().fillna(0).values],
            fmt="o", ms=4, color=color_map[f], alpha=0.85,
            elinewidth=0.4, capsize=0, ecolor=color_map[f],
            label=f"{FACILITY_RELABEL.get(f, f)}  (N={counts[f]})", zorder=4,
        )
    other = rocky_win[~fac.isin(major)]
    if len(other) > 0:
        ax.errorbar(
            other["mass_p"], other["radius_p"],
            xerr=[other["mass_err_minus"].abs().fillna(0).values,
                  other["mass_err_plus"].abs().fillna(0).values],
            yerr=[other["radius_err_minus"].abs().fillna(0).values,
                  other["radius_err_plus"].abs().fillna(0).values],
            fmt="o", ms=4, color=OTHER_COLOR, alpha=0.70,
            elinewidth=0.4, capsize=0, ecolor=OTHER_COLOR,
            label=f"Other facilities  (N={len(other)})", zorder=3,
        )

    _red_lbl = (f"Rocky threshold — {ROCKY_CURVE_PATH.name} (unshifted)"
                if abs(red_shift) < 1e-9
                else f"Rocky threshold — {ROCKY_CURVE_PATH.name} ({red_shift:+.3f} R⊕)")
    ax.plot(m_line, r_rocky,     "k--", lw=1.5, label=f"Pure-rock reference ({REF_CURVE_PATH.name})", zorder=5)
    ax.plot(m_line, r_threshold, color="crimson", lw=2.0, label=_red_lbl, zorder=5)
    ax.scatter([LHS1140B_MASS_MEARTH], [LHS1140B_RADIUS_REARTH],
               s=120, marker="*", color="dodgerblue", edgecolors="navy",
               linewidths=0.8, zorder=6, label="LHS 1140 b (anchor)")
    ax.annotate("LHS 1140 b",
                xy=(LHS1140B_MASS_MEARTH, LHS1140B_RADIUS_REARTH),
                xytext=(0.22, 0.04), textcoords="offset points",
                fontsize=8.5, color="navy")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=11)
    ax.set_ylabel(r"Radius [$R_\oplus$]", fontsize=11)
    _default_title = "Rocky threshold diagnostic — silicate curve (unshifted rocky cutoff)"
    ax.set_title(title if title is not None else _default_title, fontsize=11)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.25, linestyle="--")

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved M-R diagnostic: {out}")
    return out


def plot_threshold_curve_comparison(m_ref, r_ref, nasa_win: pd.DataFrame) -> Path:
    """Overlay the three candidate rocky thresholds: pure-rock ref.ddat shifted to LHS 1140 b (old),
    the unshifted silicate curve, and the silicate curve shifted to LHS 1140 b (current).
    """
    XLIM = (0.0, 12.0)
    YLIM = (RADIUS_LIMITS[0] - 0.05, RADIUS_LIMITS[1])
    m_line = np.linspace(XLIM[0] + 1e-3, XLIM[1], 600)

    # Pure-rock reference (ref.ddat) with its own LHS 1140 b anchor shift.
    ref_shift = float("nan")
    r_ref_pure_shifted = np.full_like(m_line, np.nan)
    if REF_CURVE_PATH.exists():
        _ref = np.loadtxt(REF_CURVE_PATH, comments="#")
        _mr, _rr = _ref[:, 0].astype(float), _ref[:, 1].astype(float)
        _o = np.argsort(_mr); _mr, _rr = _mr[_o], _rr[_o]
        ref_shift = LHS1140B_RADIUS_REARTH - float(
            radius_on_curve(LHS1140B_MASS_MEARTH, _mr, _rr)
        )
        r_ref_pure_shifted = radius_on_curve(m_line, _mr, _rr, shift=ref_shift)

    sil_anchor = LHS1140B_RADIUS_REARTH - float(
        radius_on_curve(LHS1140B_MASS_MEARTH, m_ref, r_ref)
    )
    r_sil_raw = radius_on_curve(m_line, m_ref, r_ref)
    r_sil_shifted = radius_on_curve(m_line, m_ref, r_ref, shift=sil_anchor)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    if nasa_win is not None and len(nasa_win):
        ax.scatter(nasa_win["mass_p"], nasa_win["radius_p"], s=8, c="0.85",
                   alpha=0.35, linewidths=0, zorder=1,
                   label=f"PSCompPars in-window (N={len(nasa_win):,})")

    ax.plot(m_line, r_ref_pure_shifted, color="black", lw=1.3, zorder=5,
            label=f"Pure-rock ref ({REF_CURVE_PATH.name}) → LHS 1140 b ({ref_shift:+.3f} R⊕)")
    ax.plot(m_line, r_sil_raw, color="seagreen", lw=1.3, zorder=5,
            label=f"Silicate ({ROCKY_CURVE_PATH.name}) unshifted")
    ax.plot(m_line, r_sil_shifted, color="crimson", lw=1.3, zorder=5,
            label=f"Silicate → LHS 1140 b ({sil_anchor:+.3f} R⊕)")

    ax.scatter([LHS1140B_MASS_MEARTH], [LHS1140B_RADIUS_REARTH], s=130, marker="*",
               color="dodgerblue", edgecolors="navy", linewidths=0.8, zorder=6,
               label="LHS 1140 b (anchor)")
    ax.annotate("LHS 1140 b",
                xy=(LHS1140B_MASS_MEARTH, LHS1140B_RADIUS_REARTH),
                xytext=(6, 4), textcoords="offset points", fontsize=8.5, color="navy")
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=11)
    ax.set_ylabel(r"Radius [$R_\oplus$]", fontsize=11)
    ax.set_title("Rocky threshold comparison — three candidate boundaries", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, linestyle="--")

    out = OUT_DIR / "rocky_threshold_curve_comparison.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved threshold curve comparison: {out}")
    return out


# ── Low-insolation large-radius region definition (I < 50) ───────────────────────────────────────────

# The Low-insolation large-radius region highlighted in Figure 1 and the mass-radius panels: large
# (radius > LOW_INSOLATION_REGION_RADIUS) planets receiving little insolation (I < 50).
LOW_INSOLATION_REGION_INSOL  = 50.0   # I_earth  — "low-insolation" boundary
LOW_INSOLATION_REGION_RADIUS = 1.4    # R_earth  — "large rocky" lower bound of the corner
LOW_INSOLATION_REGION_COLOR  = "#ff9ec4"  # light pink band under the silicate curve

# Insolation slices for the 1x3 mass-radius figure (title, mask, draw-corner).
# Low-insolation panels (I<10, I<50) get the pink Low-insolation-Corner band; the hot panel does not.
MR_INSOL_PANELS = [
    (r"$I < 10\,I_\oplus$",  lambda f: f < 10.0, True),
    (r"$I < 50\,I_\oplus$",  lambda f: f < 50.0, True),
    (r"$I > 50\,I_\oplus$",  lambda f: f > 50.0, False),
]


def plot_mr_insolation_panels(m_ref, r_ref, nasa_win: pd.DataFrame,
                              color_map: dict, major: list[str], counts) -> Path:
    """1x3 mass-radius panels by insolation (I<10, I<50, I>50) for the PSCompPars sample, colored by
    discovery facility, with the silicate curve and the low-insolation large-radius region (R > 1.4, I < 50) shaded.
    """
    XLIM = (0.0, 12.0)
    YLIM = (RADIUS_LIMITS[0], RADIUS_LIMITS[1])
    m_line = np.linspace(XLIM[0] + 1e-3, XLIM[1], 600)
    r_curve = radius_on_curve(m_line, m_ref, r_ref)
    # The silicate grid stops at ~11.8 M_earth; extend its last segment as a
    # power law so the curve and the pink band reach the edge of the axis.
    beyond = m_line > m_ref[-1]
    slope = np.log(r_ref[-1] / r_ref[-2]) / np.log(m_ref[-1] / m_ref[-2])
    r_curve[beyond] = r_ref[-1] * (m_line[beyond] / m_ref[-1]) ** slope
    # Gray reference = Earth-like composition (ref.ddat: 32% Fe core, 68% silicate mantle).
    r_earth = np.full_like(m_line, np.nan)
    if REF_CURVE_PATH.exists():
        _ref = np.loadtxt(REF_CURVE_PATH, comments="#")
        _ref = _ref[_ref[:, 0] > 0]
        _o = np.argsort(_ref[:, 0])
        r_earth = radius_on_curve(m_line, _ref[_o, 0], _ref[_o, 1])

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.0), sharey=True,
                             constrained_layout=True)
    fac_all = nasa_win["discovery_facility"].fillna("Unknown")

    print("\nMass-radius insolation panels (planets per panel):")
    for ax, (title, sel, draw_corner) in zip(axes, MR_INSOL_PANELS):
        panel = nasa_win[sel(nasa_win["flux_p"].to_numpy())].copy()
        fac = panel["discovery_facility"].fillna("Unknown")
        print(f"  {title:16s}  N={len(panel):3d}")

        ax.plot(m_line, r_curve, "k--", lw=1.6, zorder=5)
        ax.plot(m_line, r_earth, color="0.5", lw=2.0, zorder=5)

        for f in major:
            sub = panel[fac == f]
            if len(sub) == 0:
                continue
            ax.errorbar(
                sub["mass_p"], sub["radius_p"],
                xerr=[sub["mass_err_minus"].abs().fillna(0).values,
                      sub["mass_err_plus"].abs().fillna(0).values],
                yerr=[sub["radius_err_minus"].abs().fillna(0).values,
                      sub["radius_err_plus"].abs().fillna(0).values],
                fmt="o", ms=5, color=color_map[f], alpha=0.9,
                elinewidth=0.7, capsize=2, ecolor="0.6", zorder=4,
            )
        lhs = panel["planet_label"].str.contains(r"LHS\s*1140\s*b", case=False, na=False, regex=True)
        other = panel[~fac.isin(major)]
        for sub, ms in [(other[~lhs.loc[other.index]], 5), (other[lhs.loc[other.index]], 8)]:
            if len(sub) == 0:
                continue
            ax.errorbar(
                sub["mass_p"], sub["radius_p"],
                xerr=[sub["mass_err_minus"].abs().fillna(0).values,
                      sub["mass_err_plus"].abs().fillna(0).values],
                yerr=[sub["radius_err_minus"].abs().fillna(0).values,
                      sub["radius_err_plus"].abs().fillna(0).values],
                fmt="o", ms=ms, color=OTHER_COLOR, alpha=0.75,
                elinewidth=0.7, capsize=2, ecolor="0.6", zorder=3,
            )
        for _, row in panel[lhs].iterrows():
            ax.annotate("LHS 1140 b", xy=(row["mass_p"], row["radius_p"]),
                        xytext=(12, 0), textcoords="offset points", va="center",
                        fontsize=20, color="black", zorder=8)

        # Solar-system reference points: Earth and Venus (M = 0.815 M_earth,
        # R = 0.949 R_earth, I = 1.91 I_earth), drawn in the panels whose
        # insolation cut they satisfy.
        for name, mass, radius, flux, offset in [("Earth", 1.0, 1.0, 1.0, (12, -2)),
                                                 ("Venus", 0.815, 0.949, 1.91, (12, -20))]:
            if not sel(np.array([flux]))[0]:
                continue
            ax.plot([mass], [radius], "o", ms=8, color="black", zorder=6)
            ax.annotate(name, xy=(mass, radius), xytext=offset, textcoords="offset points",
                        va="center", fontsize=20, color="black", zorder=8)

        if draw_corner:
            ax.fill_between(
                m_line,
                LOW_INSOLATION_REGION_RADIUS,
                r_curve,
                where=np.isfinite(r_curve) & (r_curve >= LOW_INSOLATION_REGION_RADIUS),
                color=LOW_INSOLATION_REGION_COLOR,
                alpha=0.35,
                lw=0,
                zorder=1,
            )

        ax.set_xlim(*XLIM)
        ax.set_xticks(np.arange(XLIM[0], XLIM[1] + 1, 2))
        ax.set_ylim(*YLIM)
        ax.set_title(title, fontsize=26)
        ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=24)
        ax.grid(alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=21)

    axes[0].set_ylabel(r"Radius [$R_\oplus$]", fontsize=24)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[f],
               markersize=7, label=FACILITY_RELABEL.get(f, f))
        for f in major
    ]
    n_other = int(len(nasa_win) - sum(counts[f] for f in major))
    if n_other > 0:
        handles.append(Line2D([0], [0], marker="o", linestyle="", color=OTHER_COLOR,
                              markersize=7, label="Other observatories"))
    handles.append(Line2D([0], [0], color="black", lw=1.6, ls="--",
                          label=r"MgSiO$_3$ rocky curve"))
    handles.append(Line2D([0], [0], color="0.5", lw=2.0,
                          label="Earth-like rocky curve"))
    handles.append(Patch(facecolor=LOW_INSOLATION_REGION_COLOR, alpha=0.35,
                         label=r"$R>1.4\,R_\oplus$, $I<50\,I_\oplus$"))

    fig.legend(handles=handles, loc="outside right center",
               ncol=1, fontsize=20, framealpha=0.9)
    # Extra horizontal padding so the legend frame clears the last panel.
    fig.get_layout_engine().set(w_pad=0.15)

    out = OUT_DIR / "rocky_mr_insolation_3panel.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    fig.savefig(PAPER_FIG_DIR / "rocky_mr_insolation_3panel.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mass-radius insolation panels: {out}")
    print(f"Saved paper copy: {PAPER_FIG_DIR / 'rocky_mr_insolation_3panel.png'}")
    return out


# ── Standalone rocky scatter (with straight 90% line) ─────────────────────────

def plot_rocky_scatter_standalone(rocky_win: pd.DataFrame, shift: float) -> Path:
    """Insolation vs radius for in-window confirmed rocky planets, colored by
    host star type, with a straight 90% upper-bound fit line."""
    fig, ax = plt.subplots(figsize=(10, 7.5), constrained_layout=True)
    xlim = (INSOLATION_LIMITS[0], 1e4)
    ylim = (0.6, RADIUS_LIMITS[1])

    # Insolation-bin shading (matches the survival-analysis bins).
    ax.axvspan(xlim[0], LOW_INSOLATION_REGION_INSOL, color="#74add1", alpha=0.06, zorder=0)
    ax.axvspan(LOW_INSOLATION_REGION_INSOL, xlim[1], color="#fdae61", alpha=0.06, zorder=0)

    # Low-insolation large-radius region: large (R > 1.4 R_earth), low-insolation (I < 50) box.
    ax.add_patch(Rectangle(
        (INSOLATION_LIMITS[0], LOW_INSOLATION_REGION_RADIUS),
        LOW_INSOLATION_REGION_INSOL - INSOLATION_LIMITS[0], RADIUS_LIMITS[1] - LOW_INSOLATION_REGION_RADIUS,
        fill=False, edgecolor="red", lw=2.2, zorder=6,
        label=r"$R>1.4\,R_\oplus$, $I<50\,I_\oplus$",
    ))

    # Labelled planets (LHS 1140 b, Earth, Venus) use the same style as their
    # host-star group, just with a larger dot.
    labelled_ms = 6.5
    lhs_mask = rocky_win["planet_label"].str.contains(r"LHS\s*1140\s*b", case=False, na=False, regex=True)

    def draw_group(sub, color, ms, label=None):
        yerr_hi = sub["radius_err_plus"].abs().fillna(0).values
        yerr_lo = sub["radius_err_minus"].abs().fillna(0).values
        ax.errorbar(
            sub["flux_p"], sub["radius_p"], yerr=[yerr_lo, yerr_hi],
            fmt="o", ms=ms, color=color, alpha=0.85,
            elinewidth=0.9, capsize=2.5, ecolor=color,
            label=label, zorder=4,
        )

    for stype in STAR_ORDER:
        sub = rocky_win[(rocky_win["stype_clean"] == stype) & ~lhs_mask]
        if len(sub) == 0:
            continue
        draw_group(sub, STYPE_COLORS.get(stype, "gray"), 5, label=f"{stype} stars")

    # Low-insolation window is the square (S<50, R>1.4) only; the 90% upper-bound line
    # (competing definition) is intentionally omitted.
    for idx, row in rocky_win[lhs_mask].iterrows():
        draw_group(rocky_win.loc[[idx]], STYPE_COLORS.get(row["stype_clean"], "gray"), labelled_ms)
        ax.annotate("LHS 1140 b",
                    xy=(row["flux_p"], row["radius_p"]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=20, color="black", zorder=8)

    # Solar-system reference points (G host): Earth and Venus (I = 1.91 I_earth, R = 0.949 R_earth).
    for name, flux, radius, offset, ha in [("Earth", 1.0, 1.0, (0, 14), "center"),
                                           ("Venus", 1.91, 0.949, (10, -22), "left")]:
        ax.plot([flux], [radius], "o", ms=labelled_ms, color=STYPE_COLORS["G"],
                alpha=0.85, zorder=4)
        ax.annotate(name, xy=(flux, radius), xytext=offset, textcoords="offset points",
                    ha=ha, fontsize=20, color="black", zorder=8)

    _setup_axis(ax, "")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_yticks([1.0, 1.5, 2.0])
    ax.set_title("Confirmed rocky planets", fontsize=26)
    ax.set_xlabel(r"Insolation flux [$I_\oplus$]", fontsize=24)
    ax.set_ylabel(r"Planet radius [$R_\oplus$]", fontsize=24)
    ax.tick_params(labelsize=20)
    ax.set_facecolor("#fafafa")
    ax.legend(loc="lower left", ncol=5, fontsize=16, framealpha=0.90, handlelength=1.4,
              handletextpad=0.4, columnspacing=1.0, borderaxespad=0.3).set_zorder(10)

    out = OUT_DIR / "rocky_scatter_standalone.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    fig.savefig(PAPER_FIG_DIR / "rocky_scatter_standalone.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved standalone rocky scatter: {out}")
    print(f"Saved paper copy: {PAPER_FIG_DIR / 'rocky_scatter_standalone.png'}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("rocky_scatter_gaia60pc.py")
    print("=" * 70)
    print(f"Project root  : {ROOT}")
    print(f"Kepler P-Pop  : {KEPLER_PPOP_DIR}  (stacking up to {N_UNIVERSES} x kepler_catalog_i.csv)")
    print(f"TESS   P-Pop  : {TESS_PPOP_DIR}  (stacking up to {N_UNIVERSES} x tess_catalog_i.csv)")
    print(f"Output dir    : {OUT_DIR}")
    print(f"LHS 1140 b    : {LHS1140B_MASS_MEARTH} M⊕, {LHS1140B_RADIUS_REARTH} R⊕")
    print()

    m_ref, r_ref = load_rocky_reference_curve()
    shift = compute_rocky_threshold_shift(m_ref, r_ref)
    print()

    nasa_all, rocky = load_and_filter_nasa(m_ref, r_ref, shift)

    # Limit the planets used in ALL figures to the context window.
    rocky_win = restrict_science_window(
        rocky,
        insolation=INSOLATION_LIMITS,
        radius=RADIUS_LIMITS,
        stellar_types=STAR_ORDER,
    )
    nasa_win = restrict_science_window(
        nasa_all,
        insolation=INSOLATION_LIMITS,
        radius=RADIUS_LIMITS,
        stellar_types=STAR_ORDER,
    )
    rocky_win.to_csv(OUT_DIR / "rocky_pscomppars_below_threshold_in_window.csv", index=False)
    print(f"Rocky planets in context window "
          f"(R {RADIUS_LIMITS[0]}–{RADIUS_LIMITS[1]} R⊕, I < {INSOLATION_LIMITS[1]:g}): "
          f"{len(rocky):,} → {len(rocky_win):,}")
    print()

    # Shared facility color map (same colors across all figures).
    color_map, major, counts = build_facility_styles(rocky_win)

    # Figures 2 & 3 (mass-radius diagnostic, confirmed rocky scatter).
    # Main M-R diagnostic — the cutoff is now the UNSHIFTED silicate curve (shift=0).
    plot_mr_diagnostic(m_ref, r_ref, shift, nasa_win, rocky_win, color_map, major, counts)
    # Comparison: the three candidate thresholds overlaid as thin solid curves.
    plot_threshold_curve_comparison(m_ref, r_ref, nasa_win)
    plot_rocky_scatter_standalone(rocky_win, shift)

    # 1x3 mass-radius insolation panels. Facility
    # styles are built from the full in-window sample shown here (rocky + sub-Neptune).
    # Kepler and K2 are the same spacecraft, so they share one color here.
    mr_win = nasa_win.assign(discovery_facility=nasa_win["discovery_facility"].replace(
        {"Kepler": "Kepler and K2", "K2": "Kepler and K2"}))
    cm_win, major_win, counts_win = build_facility_styles(mr_win)
    # Only TESS and Kepler/K2 get their own color; everything else is "Other observatories".
    major_win = [f for f in major_win
                 if f in ("Transiting Exoplanet Survey Satellite (TESS)", "Kepler and K2")]
    plot_mr_insolation_panels(m_ref, r_ref, mr_win, cm_win, major_win, counts_win)
    print()

    if "--full" not in sys.argv:
        print("Paper figure done. Pass --full to also build the P-Pop 2x4 maps.")
        return

    catalogs = {}
    survey_specs = {
        "kepler": (
            "Kepler", KEPLER_PPOP_DIR, "kepler_catalog", prepare_kepler_catalog,
            {"mes_threshold": KEPLER_MES_THRESHOLD, "noise_scale": KEPLER_CDPP_SCALE},
        ),
        "tess": (
            "TESS", TESS_PPOP_DIR, "tess_catalog", prepare_tess_catalog,
            {"snr_threshold": TESS_SNR_THRESHOLD, "noise_scale": TESS_NOISE_SCALE},
        ),
    }
    for key, (label, directory, stem, prepare, settings) in survey_specs.items():
        files = _ppop_files(directory, stem)
        frames = []
        for run, path in enumerate(files):
            frame = pd.read_csv(path)
            frame["run"] = run
            frame["source_file"] = path.name
            frames.append(frame)
            print(f"  + {path.name}  ({len(frame):,} rows)")
        catalog = pd.concat(frames, ignore_index=True)
        print(
            f"Loaded {len(files)} {label} P-Pop universe(s): "
            f"{len(catalog):,} rows total"
        )
        catalogs[key] = prepare(
            catalog,
            insolation=INSOLATION_LIMITS,
            radius=RADIUS_LIMITS,
            stellar_types=STAR_ORDER,
            **settings,
        )
    print()

    plot_combined(
        catalogs["kepler"], catalogs["tess"], rocky_win, shift,
        color_map, major, counts,
    )
    print("\nDone.")


if __name__ == "__main__":
    try:
        # Allow unicode (⊕, ≤, …) in console output on Windows cp1252 terminals.
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
53_rv_2x4_rocky_fgkm_rvtest.py

The "RV-test" companion to script 44's transit figure, on the same FGKM
insolation-radius plane and with the same styling (smooth P-Pop background,
white contours, rocky NASA overlay colored by facility, 90% upper-bound line,
cold red window).  Now a 3x4 so the RV-test shows independently:

    Row 0 : transit-test            P(transit detected | transiting)
    Row 1 : RV-test (alone)         P(mass RV-measurable | transiting)
    Row 2 : transit + RV (joint)    P(both | transiting)  = rocky-sample selection fn

OUTPUTS
-------
    rv_test_3x4_rocky_fgkm_<inst>.png       the FGKM 3x4 overview
    rv_test_1x3_gkm_transit_<inst>.png      G/K/M, transit-test only
    rv_test_1x3_gkm_rv_<inst>.png           G/K/M, RV-test only
    rv_test_1x3_gkm_joint_<inst>.png        G/K/M, transit + RV joint

90% UPPER-BOUND LINE
--------------------
Per G/K/M panel the line is built in two explicit steps (fit_90pct_line_ols_shift):
    1. fit a straight line through the rocky NASA points by ordinary least squares
       (this sets the SLOPE);
    2. shift that line vertically up until 90% of the points lie below it
       (intercept = 90th percentile of the residuals).

SCIENCE QUESTION
----------------
Script 44 asks: is the empty region above the 90% line at low insolation a REAL
upper radius limit for cold rocky planets, or a transit selection effect?
But entering the NASA *rocky* sample requires passing TWO tests:

    transit-test : the transit must be detected            (radius)
    RV-test      : the mass must be RV-measurable          (rockiness proof)

The joint row is the full selection function of the rocky overlay sample.  If
the cold red window is bright in the transit row but dark in the joint row, the
absence of cold rocky planets above the line is explained by the RV bottleneck
— NOT evidence of a real astrophysical radius limit.  The middle row isolates
WHERE the RV-test does the damage.  Per-panel red boxes give the false-negative
probability inside the window for each test.

ROCKY BACKGROUND
----------------
Unlike script 44 (all transiting planets), the P-Pop background here is
restricted to planets BELOW the rocky threshold curve (the same unshifted
silicate curve, Hongyi-silicon.ddat, as the overlay), because at fixed radius the
RV signal depends on mass: the question is specifically whether ROCKY planets of
that radius are confirmable.

RV model: lifesim/core/rv_data.py (HARPS V-band / NIRPS J-band presets,
N=100 epochs, K/sigma_K >= 5 with sigma_K = sqrt(2/N)*sigma_RV).
--instrument best (default) takes the per-planet better of HARPS and NIRPS.

Run from repo root:
    python scripts/53_rv_2x4_rocky_fgkm_rvtest.py
    python scripts/53_rv_2x4_rocky_fgkm_rvtest.py --instrument HARPS --n-catalogs 200
"""

from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse script 44's machinery so geometry/styling/threshold match the transit
# figure exactly. We import the gaia60pc version because it carries the current
# rocky cutoff (UNSHIFTED silicate curve, Hongyi-silicon.ddat). The P-Pop
# catalogue below is intentionally the non-60pc combined sample.
S44 = _load_module("s44", ROOT / "scripts" / "44_2x4_kepler_tess_rocky-fgkm-detection_gaia60pc.py")
RVData = _load_module("rv_data", ROOT / "lifesim" / "core" / "rv_data.py").RVData

TESS_PPOP_DIR = ROOT / "run" / "tess" / "data" / "Gaia_cdpp_v1"
OUT_DIR = ROOT / "my_outputs" / "53_rv_3x4_rocky_fgkm_rvtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAR_ORDER         = S44.STAR_ORDER
INSOLATION_BINS    = S44.INSOLATION_BINS
PLANET_RADIUS_BINS = S44.PLANET_RADIUS_BINS
INSOLATION_LIMITS  = S44.INSOLATION_LIMITS
RADIUS_LIMITS      = S44.RADIUS_LIMITS
COLD_INSOLATION    = S44.COLD_INSOLATION
TESS_SNR_THRESHOLD = S44.TESS_SNR_THRESHOLD

PPOP_COLS = [
    "radius_p", "mass_p", "p_orb", "inc_p", "ecc_p",
    "radius_s", "mass_s", "teff_s", "l_sun", "distance_s", "flux_p", "stype",
    "tess_snr", "tess_observed", "tess_transiting_geometric",
    "tess_star_bright_enough", "tess_enough_transits",
]


# ── P-Pop loading (44's TESS logic + RV columns) ─────────────────────────────

def load_tess_ppop_for_rv(n_catalogs: int | None) -> pd.DataFrame:
    files = sorted(TESS_PPOP_DIR.glob("tess_catalog_*.csv"))
    if n_catalogs:
        files = files[:n_catalogs]
    if not files:
        raise FileNotFoundError(f"No tess_catalog_*.csv in {TESS_PPOP_DIR}")
    frames = [pd.read_csv(p, usecols=lambda c: c in PPOP_COLS, low_memory=False) for p in files]
    df = pd.concat(frames, ignore_index=True)

    for col in ["flux_p", "radius_p", "mass_p", "p_orb", "tess_snr"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["tess_observed", "tess_transiting_geometric",
                "tess_star_bright_enough", "tess_enough_transits"]:
        if col in df.columns:
            df[col] = S44.as_bool(df[col])
        else:
            df[col] = True

    df = S44.add_stype_clean(df)
    df = df.dropna(subset=["flux_p", "radius_p", "mass_p", "tess_snr"]).copy()
    df = S44.restrict_science_window(df)

    snr = df["tess_snr"].fillna(0.0)
    df["detected_case"] = (
        df["tess_observed"] & df["tess_transiting_geometric"]
        & df["tess_star_bright_enough"] & df["tess_enough_transits"]
        & (snr >= TESS_SNR_THRESHOLD)
    )
    df["denominator_case"] = df["tess_observed"] & df["tess_transiting_geometric"]
    print(f"Loaded {len(files)} TESS P-Pop file(s): {len(df):,} in science window "
          f"({int(df['denominator_case'].sum()):,} transiting)")
    return df


def restrict_ppop_to_rocky(df: pd.DataFrame, m_ref, r_ref, shift: float) -> pd.DataFrame:
    """Keep only P-Pop planets below the rocky threshold curve (same as the overlay)."""
    thr = S44.rocky_threshold_at_mass(df["mass_p"].to_numpy(), m_ref, r_ref, shift)
    rocky = df[(df["radius_p"].to_numpy() <= thr) & np.isfinite(thr)].copy()
    print(f"Rocky P-Pop background: {len(df):,} -> {len(rocky):,} below threshold "
          f"({int(rocky['denominator_case'].sum()):,} transiting)")
    return rocky


# ── RV test ───────────────────────────────────────────────────────────────────

def _run_single(df, instrument, snr_threshold, n_obs, mag_target):
    rv = RVData(df, source="ppop", instrument=instrument, apply_sini=True,
                snr_threshold=snr_threshold, n_obs=n_obs, validate_for_detection=False)
    cat = rv.determine_detectable()
    mag = pd.to_numeric(cat["rv_mag"], errors="coerce")
    return cat["rv_detected"].astype(bool).to_numpy() & (mag <= mag_target).to_numpy()


def add_rv_ok(df, instrument, snr_threshold, n_obs, mag_target) -> pd.DataFrame:
    """Attach rv_ok = host bright enough for RV follow-up AND K/sigma_K above threshold."""
    df = df.reset_index(drop=True)
    if instrument.lower() == "best":
        ok = (_run_single(df, "HARPS", snr_threshold, n_obs, mag_target)
              | _run_single(df, "NIRPS", snr_threshold, n_obs, mag_target))
    else:
        ok = _run_single(df, instrument, snr_threshold, n_obs, mag_target)
    df["rv_ok"] = ok
    print(f"RV-test ({instrument}): {int(df['rv_ok'].sum()):,} / {len(df):,} rocky P-Pop "
          f"planets pass the mass-measurability test")
    return df


# ── Red-window stats ──────────────────────────────────────────────────────────

def window_stats(panel: pd.DataFrame, slope: float, intercept: float, numerator: np.ndarray):
    """False-negative prob in {I<10, R above 90% line} among transiting planets."""
    f = pd.to_numeric(panel["flux_p"], errors="coerce")
    rr = pd.to_numeric(panel["radius_p"], errors="coerce")
    region = ((f < COLD_INSOLATION)
              & (rr > S44.line_radius(slope, intercept, f))
              & panel["denominator_case"].to_numpy())
    n = int(region.sum())
    if n == 0:
        return None, 0, 0
    n_pass = int((region & numerator).sum())
    return (n - n_pass) / n, n, n - n_pass


# ── Overlay styling (warm palette + light white outlines) ────────────────────

# Less blue-ish facility palette: the viridis background is blue/green, so the
# planets use warm, high-contrast colors instead of tab10's leading blues.
WARM_PALETTE = ["#e41a1c", "#ff7f00", "#e6ab02", "#f781bf",
                "#a65628", "#984ea3", "#00b8a9", "#7f0000"]


def build_warm_facility_styles(rocky_win: pd.DataFrame):
    counts = rocky_win["discovery_facility"].fillna("Unknown").value_counts()
    major = [f for f, c in counts.items() if c > S44.FACILITY_MIN_COUNT]
    color_map = {f: WARM_PALETTE[i % len(WARM_PALETTE)] for i, f in enumerate(major)}
    return color_map, major, counts


def _overlay_rocky_outlined(ax, sub: pd.DataFrame, color_map: dict, major: list[str]):
    """44-style facility overlay, but every dot and error bar gets a very light
    white outline (stroke path effect) so points stand out on any background."""
    import matplotlib.patheffects as pe
    if len(sub) == 0:
        return
    stroke = [pe.withStroke(linewidth=1.8, foreground="white", alpha=0.55)]

    yerr_hi = sub["radius_err_plus"].abs().fillna(0).values
    yerr_lo = sub["radius_err_minus"].abs().fillna(0).values
    fac = sub["discovery_facility"].fillna("Unknown").values

    def draw(mask, color, alpha, z):
        if not mask.any():
            return
        cont = ax.errorbar(
            sub["flux_p"].values[mask], sub["radius_p"].values[mask],
            yerr=[yerr_lo[mask], yerr_hi[mask]],
            fmt="o", ms=4.2, color=color, alpha=alpha,
            elinewidth=0.8, capsize=2, ecolor=color,
            markeredgecolor="white", markeredgewidth=0.45, zorder=z,
        )
        for part in list(cont[1]) + list(cont[2]):   # caps + bar line collections
            part.set_path_effects(stroke)

    for f in major:
        draw(fac == f, color_map[f], 0.9, 4)
    draw(~np.isin(fac, major), S44.OTHER_COLOR, 0.75, 3)

    lhs_mask = sub["planet_label"].str.contains(r"LHS\s*1140\s*b", case=False, na=False, regex=True)
    for _, row in sub[lhs_mask].iterrows():
        ax.scatter([row["flux_p"]], [row["radius_p"]], s=70, marker="*", color="gold",
                   edgecolors="darkred", linewidths=0.9, zorder=6)
        ax.annotate("LHS 1140 b", xy=(row["flux_p"], row["radius_p"]),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=7, color="darkred", fontweight="bold", zorder=7)


# ── 90% upper-bound line: OLS fit through the points, shifted so 90% lie below ──

def fit_90pct_line_ols_shift(flux, radius, quantile: float = 0.90):
    """Fit a straight line through the (log flux, log radius) points by ordinary
    least squares, then shift it vertically so that `quantile` of the points lie
    below it.  Returns (slope, intercept) in log10 space, or None if < 4 points.

    Two explicit steps (replaces the old pinball/quantile regression):
        1. SLOPE  = OLS slope of the line through the data.
        2. OFFSET = the q-th percentile of the residuals (y - slope*x), i.e. the
                    line is raised until `quantile` of the points sit below it.
    """
    log_f = np.log10(np.asarray(flux, dtype=float))
    log_r = np.log10(np.asarray(radius, dtype=float))
    valid = np.isfinite(log_f) & np.isfinite(log_r)
    x, y = log_f[valid], log_r[valid]
    if x.size < 4:
        return None
    slope = float(np.polyfit(x, y, 1)[0])                    # 1. linear fit through the data
    intercept = float(np.quantile(y - slope * x, quantile))  # 2. shift up: `quantile` below
    return slope, intercept


def draw_90pct_line_local(ax, flux, radius, color="black", lw=2.0, label=None, x_extent=None):
    """Fit (OLS slope + vertical 90% shift) and draw it.  Returns (slope, intercept)."""
    fit = fit_90pct_line_ols_shift(flux, radius)
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
    xs = np.logspace(np.log10(lo), np.log10(hi), 200)
    ax.plot(xs, S44.line_radius(slope, intercept, xs), color=color, lw=lw, ls="--",
            alpha=0.9, zorder=5, label=label)
    return slope, intercept


GKM_ORDER = ["G", "K", "M"]

# (display name, filename tag, numerator selector) — shared by the 3x4 and the 1x3 figures.
TESTS = [
    ("Transit-test",      "transit", lambda p: p["detected_case"].to_numpy()),
    ("RV-test",           "rv",      lambda p: p["rv_ok"].to_numpy()),
    ("Transit + RV-test", "joint",   lambda p: p["detected_case"].to_numpy() & p["rv_ok"].to_numpy()),
]


def _draw_panel(ax, p, r, test_name, stype, numerator, color_map, major,
                show_xlabel, show_ylabel):
    """Render one (test x stype) panel identically for the 3x4 and 1x3 figures.

    Returns (mesh, stat) where stat = (test_name, stype, fn, n_denom, n_missed)
    for the cold red window, or None when no line/stat could be drawn."""
    n_bg = int(p["denominator_case"].sum())
    grid, _ = S44.fraction_grid(p, pd.Series(numerator, index=p.index), p["denominator_case"])
    mesh = None
    if np.isfinite(grid).any():
        mesh = ax.pcolormesh(INSOLATION_BINS, PLANET_RADIUS_BINS, grid,
                             shading="auto", vmin=0, vmax=1, cmap=S44.CMAP_DETECTED)
        S44._add_contours(ax, grid)
    else:
        ax.text(0.5, 0.5, "no P-Pop data", transform=ax.transAxes,
                ha="center", va="center", color="0.5")

    _overlay_rocky_outlined(ax, r, color_map, major)

    stat = None
    if stype in ("G", "K", "M"):
        fit = draw_90pct_line_local(ax, r["flux_p"].values, r["radius_p"].values,
                                    color="black", lw=2.0, x_extent=INSOLATION_LIMITS)
        if fit is not None:
            slope, intercept = fit
            xreg = np.logspace(np.log10(INSOLATION_LIMITS[0]),
                               np.log10(COLD_INSOLATION), 100)
            ax.fill_between(xreg, S44.line_radius(slope, intercept, xreg),
                            RADIUS_LIMITS[1], color="red", alpha=0.12, zorder=1.5, lw=0)
            fn, n_denom, n_missed = window_stats(p, slope, intercept, numerator)
            if fn is not None:
                ax.text(0.03, 0.97,
                        f"P(false neg | I<10, R>fit) = {fn:.0%}\n"
                        f"({n_missed}/{n_denom} rocky P-Pop missed)",
                        transform=ax.transAxes, va="top", ha="left", fontsize=7,
                        color="darkred",
                        bbox=dict(boxstyle="round", fc="white", ec="darkred", alpha=0.85),
                        zorder=8)
                stat = (test_name, stype, fn, n_denom, n_missed)

    S44._setup_axis(
        ax,
        f"{test_name} — {stype} stars\n"
        f"N_rockyPPop(transiting) = {n_bg:,}   ·   N_rocky_NASA = {len(r)}",
        show_xlabel=show_xlabel, show_ylabel=show_ylabel,
    )
    return mesh, stat


def _legend_handles(rocky_win, color_map, major, counts):
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
        Line2D([0], [0], marker="*", linestyle="", color="gold",
               markeredgecolor="darkred", markersize=12, label="LHS 1140 b (anchor)"),
        Line2D([0], [0], color="black", lw=2.0, ls="--",
               label="90% upper-bound (OLS fit, shifted so 90% below; G/K/M)"),
        Patch(facecolor="red", alpha=0.12, label="Cold red window (I<10, R>fit)"),
    ]
    return handles


def plot_test_1x3(test_name: str, tag: str, numerator_fn, ppop, rocky_win,
                  color_map, major, counts, args) -> Path:
    """One 1x3 (G/K/M) figure for a single test, using the new OLS+shift 90% line."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), sharex=True, sharey=True,
                             constrained_layout=True)
    mesh = None
    print(f"\n{test_name} — 1x3 G/K/M:")
    for j, stype in enumerate(GKM_ORDER):
        p = ppop[ppop["stype_clean"] == stype].copy()
        r = rocky_win[rocky_win["stype_clean"] == stype].copy()
        m, stat = _draw_panel(axes[j], p, r, test_name, stype, numerator_fn(p),
                              color_map, major, show_xlabel=True, show_ylabel=(j == 0))
        if m is not None:
            mesh = m
        if stat is not None:
            print(f"  {test_name:18s} {stype}: {stat[2]:.1%}  ({stat[4]}/{stat[3]})")

    fig.legend(handles=_legend_handles(rocky_win, color_map, major, counts),
               loc="outside lower center", ncol=4, fontsize=8.5, framealpha=0.9,
               title="Rocky NASA planets — colored by discovery facility (telescope/mission)",
               title_fontsize=9.5)
    if mesh is not None:
        fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Pass fraction (rocky P-Pop)",
                     shrink=0.7, pad=0.01)
    fig.suptitle(
        f"{test_name} — rocky P-Pop selection function (G/K/M)\n"
        "90% upper-bound = straight OLS fit through the rocky points, shifted up so 90% lie below  |  "
        f"RV: {args.instrument}, K/σ_K ≥ {args.snr_threshold}, N={args.n_obs}, band-mag ≤ {args.mag_target}",
        fontsize=12.5,
    )
    out = OUT_DIR / f"rv_test_1x3_gkm_{tag}_{args.instrument.lower()}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ── Figure ────────────────────────────────────────────────────────────────────

def plot_rv_test(ppop: pd.DataFrame, rocky_win: pd.DataFrame, shift: float,
                 color_map: dict, major: list[str], counts, args) -> Path:
    rows = [
        ("Transit-test",      lambda p: p["detected_case"].to_numpy()),
        ("RV-test",           lambda p: p["rv_ok"].to_numpy()),
        ("Transit + RV-test", lambda p: p["detected_case"].to_numpy() & p["rv_ok"].to_numpy()),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(24, 15.5), sharex=True, sharey=True,
                             constrained_layout=True)
    mesh = None
    stats_lines = []

    print("\nCold red-window false-negative probabilities (I<10, radius above 90% fit):")
    for i, (test_name, numerator_fn) in enumerate(rows):
        for j, stype in enumerate(STAR_ORDER):
            ax = axes[i, j]
            p = ppop[ppop["stype_clean"] == stype].copy()
            r = rocky_win[rocky_win["stype_clean"] == stype].copy()

            n_bg = int(p["denominator_case"].sum())
            numerator = numerator_fn(p)
            grid, _ = S44.fraction_grid(p, pd.Series(numerator, index=p.index),
                                        p["denominator_case"])
            if np.isfinite(grid).any():
                mesh = ax.pcolormesh(INSOLATION_BINS, PLANET_RADIUS_BINS, grid,
                                     shading="auto", vmin=0, vmax=1, cmap=S44.CMAP_DETECTED)
                S44._add_contours(ax, grid)
            else:
                ax.text(0.5, 0.5, "no P-Pop data", transform=ax.transAxes,
                        ha="center", va="center", color="0.5")

            _overlay_rocky_outlined(ax, r, color_map, major)

            if stype in ("G", "K", "M"):
                fit = draw_90pct_line_local(ax, r["flux_p"].values, r["radius_p"].values,
                                            color="black", lw=2.0, x_extent=INSOLATION_LIMITS)
                if fit is not None:
                    slope, intercept = fit
                    xreg = np.logspace(np.log10(INSOLATION_LIMITS[0]),
                                       np.log10(COLD_INSOLATION), 100)
                    ax.fill_between(xreg, S44.line_radius(slope, intercept, xreg),
                                    RADIUS_LIMITS[1], color="red", alpha=0.12,
                                    zorder=1.5, lw=0)
                    fn, n_denom, n_missed = window_stats(p, slope, intercept, numerator)
                    if fn is not None:
                        ax.text(0.03, 0.97,
                                f"P(false neg | I<10, R>fit) = {fn:.0%}\n"
                                f"({n_missed}/{n_denom} rocky P-Pop missed)",
                                transform=ax.transAxes, va="top", ha="left",
                                fontsize=7, color="darkred",
                                bbox=dict(boxstyle="round", fc="white",
                                          ec="darkred", alpha=0.85), zorder=8)
                        print(f"  {test_name:18s} {stype}: {fn:.1%}  ({n_missed}/{n_denom})")
                        stats_lines.append((test_name, stype, fn, n_denom, n_missed))

            S44._setup_axis(
                ax,
                f"{test_name} — {stype} stars\n"
                f"N_rockyPPop(transiting) = {n_bg:,}   ·   N_rocky_NASA = {len(r)}",
                show_xlabel=(i == 2), show_ylabel=(j == 0),
            )

    # Shared legend (44 style).
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
        Line2D([0], [0], marker="*", linestyle="", color="gold",
               markeredgecolor="darkred", markersize=12, label="LHS 1140 b (anchor)"),
        Line2D([0], [0], color="black", lw=2.0, ls="--",
               label="90% upper-bound fit (≈90% of planets below; G/K/M)"),
        Patch(facecolor="red", alpha=0.12, label="Cold red window (I<10, R>fit)"),
    ]
    fig.legend(handles=handles, loc="outside lower center",
               ncol=min(len(handles), 4), fontsize=9, framealpha=0.9,
               title="Rocky NASA planets — colored by discovery facility (telescope/mission)",
               title_fontsize=10)
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="Pass fraction (rocky P-Pop)",
                 shrink=0.6, pad=0.01)
    fig.suptitle(
        "Transit-test (top) · RV-test alone (middle) · Transit + RV joint (bottom) — "
        "rocky P-Pop selection function, FGKM\n"
        f"Background = rocky P-Pop (below the unshifted silicate threshold, Hongyi-silicon.ddat); "
        f"RV-test = mass RV-measurable "
        f"({args.instrument}: K/σ_K ≥ {args.snr_threshold}, N={args.n_obs}, band-mag ≤ {args.mag_target})\n"
        "If the cold red window is bright on top but dark in the joint row, the missing cold "
        "rocky planets are a selection effect — not a real radius limit",
        fontsize=13,
    )

    out = OUT_DIR / f"rv_test_3x4_rocky_fgkm_{args.instrument.lower()}.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")
    return stats_lines


def write_summary(stats_lines, args):
    lines = [
        "=" * 78,
        "RV-TEST ON THE COLD RED WINDOW — rocky P-Pop selection function",
        "=" * 78,
        f"RV: {args.instrument}, K/σ_K ≥ {args.snr_threshold}, N_obs={args.n_obs}, "
        f"band-mag ≤ {args.mag_target}  |  rocky = below unshifted silicate threshold",
        "",
        f"{'test':<20}{'stype':<7}{'P(false neg)':>14}{'missed/N':>12}",
    ]
    for test, st, fn, n, miss in stats_lines:
        lines.append(f"{test:<20}{st:<7}{fn:>13.0%}{f'{miss}/{n}':>12}")
    lines += [
        "",
        "Reading: row-1 false-negative ≈ 100% in the red window means a cold rocky",
        "planet there could (almost) never enter the NASA mass-confirmed sample —",
        "the apparent upper radius limit at low insolation is consistent with pure",
        "selection (transit x RV), and is NOT evidence of a real astrophysical bound.",
        "Wherever row 1 retains sensitivity but NASA is still empty, a real limit",
        "remains possible — that is the discriminating region.",
        "=" * 78,
    ]
    text = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-catalogs", type=int, default=None,
                    help="P-Pop catalogs to load (default: all).")
    ap.add_argument("--instrument", default="best", help="HARPS | NIRPS | best")
    ap.add_argument("--snr-threshold", type=float, default=5.0)
    ap.add_argument("--n-obs", type=int, default=100)
    ap.add_argument("--mag-target", type=float, default=12.0)
    args = ap.parse_args()

    print("=" * 70)
    print("53_rv_2x4_rocky_fgkm_rvtest.py — RV-test companion to script 44")
    print("=" * 70)

    m_ref, r_ref = S44.load_rocky_reference_curve()
    shift = S44.compute_rocky_threshold_shift(m_ref, r_ref)
    nasa_all, rocky = S44.load_and_filter_nasa(m_ref, r_ref, shift)
    rocky_win = S44.restrict_to_window(rocky)
    print(f"Rocky NASA planets in window: {len(rocky_win):,}")
    color_map, major, counts = build_warm_facility_styles(rocky_win)

    ppop = load_tess_ppop_for_rv(args.n_catalogs)
    ppop = restrict_ppop_to_rocky(ppop, m_ref, r_ref, shift)
    ppop = add_rv_ok(ppop, args.instrument, args.snr_threshold, args.n_obs, args.mag_target)

    stats = plot_rv_test(ppop, rocky_win, shift, color_map, major, counts, args)

    # One separate 1x3 (G/K/M) figure per test: transit, RV, combined.
    for test_name, tag, numerator_fn in TESTS:
        plot_test_1x3(test_name, tag, numerator_fn, ppop, rocky_win,
                      color_map, major, counts, args)

    write_summary(stats, args)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)

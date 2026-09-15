"""Which rocky M-R relation (Chen & Kipping 2017, Otegi 2020, Edmondson 2023, Müller 2024), imposed
on the flat universe, best matches NASA's volatile ("puffy") fraction? Makes the 2x4 grids and the
paper's Otegi panels. Run: python plotting/scripts/analysis/multi/flat_rocky_mr_vs_nasa.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

from tools.paths import LIFESIM_OUTER_DIR, ANALYSIS_DIR, PAPER_FIGURES_DIR
ROOT = Path(LIFESIM_OUTER_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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

from run.flat_universe.uniform_generator import generate_flat_catalog
from run.ppop.flat_detect import run_kepler, run_rv_best
from plotting.scripts.analysis.multi import puffy_cuts_flat as puffy_cuts

puffy_cuts.N_REPEATS = 4000

OUT_DIR = os.path.join(ANALYSIS_DIR, "flat_rocky_mr_vs_nasa")
PAPER_FIG_DIR = Path(PAPER_FIGURES_DIR)

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
})
FLAT_N = 150000
SEED = 0
MR_SCATTER_DEX = 0.15           # common log-normal mass scatter around each rocky relation (tunable)

# (name, equation, applies-over, {mr_C, mr_beta})   R = C·M^β  (R in R⊕, M in M⊕)
RELATIONS = [
    ("Chen & Kipping 2017", r"$R=1.01\,M^{0.28}$", r"M < 2.04 $M_\oplus$ (Terran)", dict(mr_C=1.01, mr_beta=0.28)),
    ("Otegi et al. 2020",   r"$R=1.03\,M^{0.29}$", r"rocky branch",                dict(mr_C=1.03, mr_beta=0.29)),
    ("Edmondson et al. 2023", r"$R=0.99\,M^{0.34}$", r"M $\lesssim$ 4-5 $M_\oplus$", dict(mr_C=0.99, mr_beta=0.34)),
    ("Müller et al. 2024",  r"$R=1.02\,M^{0.27}$", r"M < 4.37 $M_\oplus$",         dict(mr_C=1.02, mr_beta=0.27)),
]


OTEGI_VOLATILE = dict(mr_C=0.70, mr_beta=0.63)   # Otegi et al. 2020 volatile-rich branch
SUB_NEPTUNE_FRAC_SD = 0.20       # fractional radius width around the volatile-rich curve
SUPER_EARTH_FRAC_SD = 0.20       # ... and around the silicate line


def otegi_volatile_radius(mass):
    return OTEGI_VOLATILE["mr_C"] * mass ** OTEGI_VOLATILE["mr_beta"]


def build_arrays(rel_kw, m_sil, r_sil, two_populations=False, rng=None):
    """Detected flat-universe pool for one mass-radius relation. With two_populations, radii are
    redrawn: sub-Neptunes around Otegi's volatile-rich relation, rocky super-Earths (M > 2) around the
    silicate line (SUB_NEPTUNE_FRAC_SD / SUPER_EARTH_FRAC_SD); planets leaving 0.5-2.2 R_earth drop.
    """
    cat = generate_flat_catalog(FLAT_N, seed=SEED, mass_model="powerlaw",
                                mass_scatter_dex=MR_SCATTER_DEX, **rel_kw)
    r = pd.to_numeric(cat["radius_p"], errors="coerce")
    m = pd.to_numeric(cat["mass_p"], errors="coerce")
    f = pd.to_numeric(cat["flux_p"], errors="coerce")
    keep = (r.between(puffy_cuts.BOX["r_lo"], puffy_cuts.BOX["r_hi"]) & m.between(puffy_cuts.BOX["m_lo"], puffy_cuts.BOX["m_hi"])
            & (f.isna() | f.between(puffy_cuts.BOX["f_lo"], puffy_cuts.BOX["f_hi"])))
    cat = cat[keep].copy()
    mass = pd.to_numeric(cat["mass_p"], errors="coerce").to_numpy(float)
    radius = pd.to_numeric(cat["radius_p"], errors="coerce").to_numpy(float).copy()
    flux = pd.to_numeric(cat["flux_p"], errors="coerce").to_numpy(float)
    puffy = radius > np.interp(mass, m_sil, r_sil)
    if two_populations:
        se = (~puffy) & (mass > puffy_cuts.MASS_THRESHOLD)
        mu = np.where(se, np.interp(mass, m_sil, r_sil), otegi_volatile_radius(mass))
        frac_sd = np.where(se, SUPER_EARTH_FRAC_SD, SUB_NEPTUNE_FRAC_SD)
        radius = mu * (1.0 + frac_sd * rng.standard_normal(mass.size))
        in_box = (radius >= puffy_cuts.BOX["r_lo"]) & (radius <= puffy_cuts.BOX["r_hi"])
        cat["radius_p"] = radius
        cat = cat[in_box].copy()
        mass, radius, flux = mass[in_box], radius[in_box], flux[in_box]
        puffy = radius > np.interp(mass, m_sil, r_sil)
    td = run_kepler(cat)["detected"].to_numpy(bool)
    rd = run_rv_best(cat, mag_target=puffy_cuts.RV_MAG_TARGET)["detected"].to_numpy(bool)
    return mass, radius, flux, puffy, td & rd


def noised_scatter_AB(arrays, cut, rng, n_plot=400):
    """One detected + noised draw (with the cut applied); split into flat-A-kept vs dropped-by-A."""
    mass, radius, flux, puffy, det = arrays
    keep = det.copy()
    if cut.get("insol_max"):
        keep = keep & (flux < cut["insol_max"])
    idx = np.flatnonzero(keep)
    mo = mass[idx] * np.exp(rng.normal(0, puffy_cuts.MASS_FRAC_ERR, idx.size))
    ro = radius[idx] * np.exp(rng.normal(0, puffy_cuts.RAD_FRAC_ERR, idx.size))
    tmass, tpuffy = mass[idx], puffy[idx]
    if cut.get("mass_min"):
        k = mo > cut["mass_min"]
        mo, ro, tmass, tpuffy = mo[k], ro[k], tmass[k], tpuffy[k]
    dropped = (~tpuffy) & (tmass > puffy_cuts.MASS_THRESHOLD)          # A drops these (true rocky, true M>2)
    if mo.size > n_plot:
        j = rng.choice(mo.size, n_plot, replace=False)
        mo, ro, dropped = mo[j], ro[j], dropped[j]
    return mo, ro, dropped


CUTS = [("all (no cut)", {}, "flat_rocky_mr_relations_2x4.png"),
        ("insol<50 I⊕", dict(insol_max=50.0), "flat_rocky_mr_relations_2x4_s50.png"),
        ("mass>2 M⊕ & insol<50 I⊕ (cold super-Earth cut)",
         dict(mass_min=2.0, insol_max=50.0), "flat_rocky_mr_relations_2x4_cold_corner.png")]

# The 2x2 paper figure shows the default relation (Otegi) alone, before/after the cold cut.
OTEGI_2X2_CUTS = [("Full detected sample", {}),
                  ("Cold super-Earth cut", dict(mass_min=2.0, insol_max=50.0))]


def nasa_cut(nasa, cut):
    m, r = nasa["m"], nasa["r"]
    me1, me2, re1, re2 = nasa["me1"], nasa["me2"], nasa["re1"], nasa["re2"]
    if cut.get("insol_max"):
        sel = nasa["ins"] < cut["insol_max"]
        m, r, me1, me2, re1, re2 = m[sel], r[sel], me1[sel], me2[sel], re1[sel], re2[sel]
    if cut.get("mass_min"):
        sel = m > cut["mass_min"]
        m, r, me1, me2, re1, re2 = m[sel], r[sel], me1[sel], me2[sel], re1[sel], re2[sel]
    return m, r, me1, me2, re1, re2


def true_sample(arrays, cut, n, rng, above_line_only=False):
    """n detected planets at their TRUE masses and radii, with the cut applied to
    true values. above_line_only keeps only planets truly above the silicate
    line (universe A's cut)."""
    mass, radius, flux, puffy, det = arrays
    keep = det.copy()
    if cut.get("insol_max"):
        keep &= flux < cut["insol_max"]
    if cut.get("mass_min"):
        keep &= mass > cut["mass_min"]
    if above_line_only:
        keep &= puffy
    idx = np.flatnonzero(keep)
    if idx.size > n:
        idx = rng.choice(idx, n, replace=False)
    return mass[idx], radius[idx]


def _frac_err_bars(x, frac_sd):
    """1-sigma bars for a log-normal fractional error: [lower, upper] offsets."""
    return np.array([x * (1 - np.exp(-frac_sd)), x * (np.exp(frac_sd) - 1)])


SCATTER_LABELS = ("Escape-only (kept)", "Primordial-rocky: rocky super-Earths (M>2)")


def extend_silicate(m_sil, r_sil, m_lo, m_hi, n=400):
    """The silicate curve on [m_lo, m_hi], continued past its tabulated ends
    as power laws with the slopes of its first and last segments."""
    m = np.linspace(m_lo, m_hi, n)
    lm, lr = np.log(m_sil), np.log(r_sil)
    r = np.exp(np.interp(np.log(m), lm, lr))
    lo, hi = m < m_sil[0], m > m_sil[-1]
    s_lo = (lr[1] - lr[0]) / (lm[1] - lm[0])
    s_hi = (lr[-1] - lr[-2]) / (lm[-1] - lm[-2])
    r[lo] = r_sil[0] * (m[lo] / m_sil[0]) ** s_lo
    r[hi] = r_sil[-1] * (m[hi] / m_sil[-1]) ** s_hi
    return m, r


def _draw_scatter(ax, arr, cut, nasa, m_sil, r_sil, rng, title,
                  labels=SCATTER_LABELS, true_values=False, sil_range=None):
    """Top-row panel: one detected, noise-perturbed mass-radius draw. With
    true_values, the planets are drawn at their true masses and radii with the
    simulated measurement errors as error bars instead."""
    nmc, nrc, nme1, nme2, nre1, nre2 = nasa_cut(nasa, cut)
    # sil_range=(lo, hi) draws the curve across that whole mass range.
    m_c, r_c = extend_silicate(m_sil, r_sil, *sil_range) if sil_range else (m_sil, r_sil)
    ax.fill_between(m_c, r_c, 2.6, color="0.965", zorder=0)
    ax.plot(m_c, r_c, "k-", lw=1.2, zorder=6, label="silicate line")
    if true_values:
        # One illustrative survey per universe: blue (A) from the planets that
        # survive its cut, orange (B) from every planet.
        for above_only, n, colour, lbl, z in [
                (True, N_SURVEY_BLUE, "tab:blue", labels[0], 4),
                (False, N_SURVEY_ORANGE, "tab:orange", labels[1], 3)]:
            mt, rt = true_sample(arr, cut, n, rng, above_line_only=above_only)
            ax.errorbar(mt, rt, xerr=_frac_err_bars(mt, puffy_cuts.MASS_FRAC_ERR),
                        yerr=_frac_err_bars(rt, puffy_cuts.RAD_FRAC_ERR),
                        fmt="o", ms=4, color=colour, alpha=0.6, elinewidth=0.6,
                        capsize=0, zorder=z, label=lbl)
    else:
        mo, ro, dropped = noised_scatter_AB(arr, cut, rng)
        ax.scatter(mo[~dropped], ro[~dropped], s=15, color="tab:blue", alpha=0.45, lw=0,
                   zorder=3, label=labels[0])
        ax.scatter(mo[dropped], ro[dropped], s=15, color="tab:orange", alpha=0.5, lw=0,
                   zorder=4, label=labels[1])
    ax.errorbar(nmc, nrc, xerr=np.array([nme2, nme1]), yerr=np.array([nre2, nre1]),
                fmt="o", mfc="none", mec="k", ecolor="k", ms=5, mew=1.0,
                elinewidth=0.6, capsize=1.5, alpha=0.8, zorder=5,
                label=f"NASA (N={nmc.size})")
    ax.set_xlim(0, 13); ax.set_ylim(0.5, 2.4)
    ax.set_title(title)
    ax.grid(alpha=0.2); ax.legend(fontsize=10, loc="lower right", framealpha=0.9)
    ax.text(0.2, 2.3, "VOLATILE (above)", fontsize=11, color="0.4", va="top")
    ax.text(4.0, 0.6, "ROCKY (below)", fontsize=11, color="0.4")
    ax.set_xlabel(r"planet mass [$M_\oplus$]")
    ax.set_ylabel(r"planet radius [$R_\oplus$]")


# Planets per simulated survey in the 1x2: blue ("Sub-Neptunes only") and orange
# ("Sub-Neptunes and super-Earths"). The left panel shows one survey of each size.
N_SURVEY_BLUE = 50
N_SURVEY_ORANGE = 100


def mc_universe_blue_cut(arrays, drop, cut, m_sil, r_sil, rng):
    """Volatile-fraction Monte Carlo for the 1x2's two universes from one pool: each of puffy_cuts.N_REPEATS
    surveys draws N_SURVEY_BLUE (blue: truly above the silicate line) or N_SURVEY_ORANGE (orange: whole
    pool) planets that pass the cut after measurement noise.
    """
    mass, radius, flux, puffy, det = arrays
    n = N_SURVEY_BLUE if drop else N_SURVEY_ORANGE
    pool = det.copy()
    if cut.get("insol_max"):
        pool &= flux < cut["insol_max"]
    if drop:
        pool &= puffy
    idx = np.flatnonzero(pool)
    mass_min = cut.get("mass_min") or 0.0
    out = np.full(puffy_cuts.N_REPEATS, np.nan)
    for i in range(puffy_cuts.N_REPEATS):
        m_obs, r_obs = np.empty(0), np.empty(0)
        while m_obs.size < n:
            s = rng.choice(idx, 4 * n)
            mo = mass[s] * np.exp(rng.normal(0, puffy_cuts.MASS_FRAC_ERR, s.size))
            ro = radius[s] * np.exp(rng.normal(0, puffy_cuts.RAD_FRAC_ERR, s.size))
            k = mo > mass_min
            m_obs, r_obs = np.append(m_obs, mo[k]), np.append(r_obs, ro[k])
        out[i] = puffy_cuts.puffy_frac(m_obs[:n], r_obs[:n], m_sil, r_sil)
    return out, float(n)


def _draw_bells(ax, arr, cut, nasa, m_sil, r_sil, rng, tag=""):
    """Bottom-row panel: COUNT histograms of the volatile fraction over the MC draws.
    y = number of the N_REPEATS draws that landed in each bin; N_p = mean planets/draw."""
    nv, n_nasa = puffy_cuts.mc_nasa(nasa, cut, m_sil, r_sil, rng)
    n_mu, n_sd = nv.mean(), nv.std()
    sA, nA = puffy_cuts.mc_universe(arr, True, cut, m_sil, r_sil, rng)
    sB, nB = puffy_cuts.mc_universe(arr, False, cut, m_sil, r_sil, rng)
    all_bell = [b for b in (nv, sA, sB) if b.size]
    cat = np.concatenate(all_bell)
    lo, hi = cat.min(), cat.max(); pad = 0.05 * (hi - lo)
    gx = np.linspace(lo - pad, hi + pad, 400)
    # fixed 0.04-wide bins on the k/25 grid, identical across all figures: NASA's
    # fraction is discrete (same ~25 planets each draw), finer data-driven bins alias it
    edges = np.arange(-0.02, 1.02 + 1e-9, 0.04)
    bw = edges[1] - edges[0]
    # Taller of the bars and the fitted curves, so no curve runs off the top.
    y_max = max(max(np.histogram(b, bins=edges)[0].max(),
                    puffy_cuts.gauss(b.mean(), b.mean(), b.std()) * b.size * bw) for b in all_bell)
    for lbl, s, ne, colour in [("Escape-only", sA, nA, "tab:orange"),
                               ("Primordial-rocky", sB, nB, "tab:blue")]:
        if s.size == 0:
            continue
        tens = abs(s.mean() - n_mu) / np.sqrt(s.std() ** 2 + n_sd ** 2)
        ax.hist(s, bins=edges, color=colour, alpha=0.30)
        ax.plot(gx, puffy_cuts.gauss(gx, s.mean(), s.std()) * s.size * bw, color=colour, lw=2.0,
                label=f"{lbl}: $\\mu$={s.mean():.2f} $\\sigma$={s.std():.3f} "
                      f"({tens:.1f}$\\sigma$), N$_p$={ne:.0f}")
        if tag:
            print(f"    {tag} {lbl}: mu={s.mean():.3f} sd={s.std():.3f} "
                  f"tension={tens:.1f}sigma N_planets={ne:.0f} N_draws={s.size}")
    ax.hist(nv, bins=edges, color="tab:green", alpha=0.34)
    ax.plot(gx, puffy_cuts.gauss(gx, n_mu, n_sd) * nv.size * bw, color="tab:green", lw=2.4,
            label=f"NASA: $\\mu$={n_mu:.2f} $\\sigma$={n_sd:.3f}, N$_p$={n_nasa}")
    ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(0, y_max * 1.15)
    ax.grid(alpha=0.2); ax.legend(fontsize=9, loc="upper left")
    ax.set_xlabel("volatile fraction")
    ax.set_ylabel(f"number of MC draws (of {puffy_cuts.N_REPEATS:,})")


def make_figure(cut_label, cut, fname, pools, nasa, m_sil, r_sil, rng):
    print(f"\n--> [{cut_label}]")
    fig, axes = plt.subplots(2, 4, figsize=(23, 11))
    for ci, (name, eq, applies, arr) in enumerate(pools):
        _draw_scatter(axes[0, ci], arr, cut, nasa, m_sil, r_sil, rng, f"{name}\n{eq}")
        _draw_bells(axes[1, ci], arr, cut, nasa, m_sil, r_sil, rng, tag=f"[{cut_label}] {name}")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, fname)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    if fname == "flat_rocky_mr_relations_2x4_cold_corner.png":
        PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PAPER_FIG_DIR / fname, dpi=170, bbox_inches="tight")
        print(f"--> Saved paper copy: {PAPER_FIG_DIR / fname}")
    plt.close(fig)
    print(f"--> Saved: {out_png}")


def make_otegi_2x2(arr, nasa, m_sil, r_sil, rng):
    """Appendix figure: the default (Otegi) relation only — mass-radius plane on top,
    volatile-fraction count histograms below, before (left) and after (right) the cut."""
    print("\n--> Otegi 2x2 (before/after the cold super-Earth cut):")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11))
    for ci, (cut_label, cut) in enumerate(OTEGI_2X2_CUTS):
        _draw_scatter(axes[0, ci], arr, cut, nasa, m_sil, r_sil, rng, cut_label)
        _draw_bells(axes[1, ci], arr, cut, nasa, m_sil, r_sil, rng, tag=f"[{cut_label}]")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "flat_otegi_2x2_before_after.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_DIR / "flat_otegi_2x2_before_after.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out_png}")
    print(f"--> Saved paper copy: {PAPER_FIG_DIR / 'flat_otegi_2x2_before_after.png'}")


def make_otegi_2x1(arr, nasa, m_sil, r_sil, rng):
    """Main-text figure: Otegi relation, cold super-Earth cut only — mass-radius (top)
    + volatile-fraction count histogram (bottom), stacked for single-column width."""
    print("\n--> Otegi 2x1 (cold super-Earth cut only):")
    cut_label, cut = OTEGI_2X2_CUTS[1]      # the 'after' column (cold super-Earth cut)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 10.8))
    _draw_scatter(axes[0], arr, cut, nasa, m_sil, r_sil, rng, cut_label)
    _draw_bells(axes[1], arr, cut, nasa, m_sil, r_sil, rng, tag=f"[{cut_label}]")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "flat_otegi_2x1_cold_cut.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_DIR / "flat_otegi_2x1_cold_cut.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out_png}")
    print(f"--> Saved paper copy: {PAPER_FIG_DIR / 'flat_otegi_2x1_cold_cut.png'}")


def _draw_density_1x2(ax, arr, cut, nasa, m_sil, r_sil, rng, tag=""):
    """Right panel of the Otegi 1x2: probability densities of the volatile
    fraction. Blue / orange: histograms of the 50- / 100-planet surveys with
    their fitted normals. NASA: its fitted normal only, filled (its 27 planets
    are the same in every draw, so only its mean and spread matter)."""
    nv, _ = puffy_cuts.mc_nasa(nasa, cut, m_sil, r_sil, rng)
    n_mu, n_sd = nv.mean(), nv.std()
    sA, _ = mc_universe_blue_cut(arr, True, cut, m_sil, r_sil, rng)
    sB, _ = mc_universe_blue_cut(arr, False, cut, m_sil, r_sil, rng)
    cat = np.concatenate([nv, sA, sB])
    lo, hi = cat.min(), cat.max(); pad = 0.05 * (hi - lo)
    gx = np.linspace(lo - pad, hi + pad, 400)
    # Surveys of 50 / 100 planets give fractions on a 0.02 / 0.01 grid, so 0.02 is
    # the narrowest bin that leaves no blue bar empty; edges at half-hundredths
    # keep every grid value off a bin edge.
    edges = np.arange(-0.005, 1.02 + 1e-9, 0.02)
    y_max = 0.0
    for lbl, s, colour in [("Sub-Neptunes only", sA, "tab:blue"),
                           ("Sub-Neptunes and super-Earths", sB, "tab:orange")]:
        heights, _, _ = ax.hist(s, bins=edges, density=True, color=colour, alpha=0.30)
        pdf = puffy_cuts.gauss(gx, s.mean(), s.std())
        ax.plot(gx, pdf, color=colour, lw=2.0,
                label=f"{lbl}: $\\mu$={s.mean():.2f} $\\sigma$={s.std():.3f}")
        y_max = max(y_max, heights.max(), pdf.max())
        tens = abs(s.mean() - n_mu) / np.sqrt(s.std() ** 2 + n_sd ** 2)
        print(f"    {tag} {lbl}: mu={s.mean():.3f} sd={s.std():.3f} "
              f"tension={tens:.1f}sigma N_draws={s.size}")
    pdf = puffy_cuts.gauss(gx, n_mu, n_sd)
    ax.fill_between(gx, pdf, color="tab:green", alpha=0.30, lw=0)
    ax.plot(gx, pdf, color="tab:green", lw=2.4,
            label=f"Measured exoplanets: $\\mu$={n_mu:.2f} $\\sigma$={n_sd:.3f}")
    y_max = max(y_max, pdf.max())
    ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(0, y_max * 1.4)   # headroom for the legend
    ax.legend(fontsize=16, loc="upper left")
    ax.set_xlabel("Volatile fraction")
    ax.set_ylabel("Probability density")


def make_otegi_1x2(nasa, m_sil, r_sil, rng):
    """Otegi mass-radius draw (left) beside its volatile-fraction histograms (right), cold cut.
    Orange = whole pool; blue = pool minus planets truly on/below the silicate line. Left shows true
    values with simulated error bars; histograms use noisy values, so some blue can fall below the line.
    """
    print("\n--> Otegi 1x2 (cold super-Earth cut; two-population radii):")
    cut_label, cut = OTEGI_2X2_CUTS[1]
    otegi_kw = next(kw for name, eq, applies, kw in RELATIONS if "Otegi" in name)
    arr = build_arrays(otegi_kw, m_sil, r_sil, two_populations=True,
                       rng=np.random.default_rng(SEED + 1))
    fig, axes = plt.subplots(1, 2, figsize=(17.0, 7.5))
    ax_hist, ax_mr = axes      # histograms on the left, the illustration on the right
    # The mass-radius draw runs first so the histogram draws use the same random
    # stream as before the panels were swapped.
    _draw_scatter(ax_mr, arr, cut, nasa, m_sil, r_sil, rng, "",
                  labels=("Sub-Neptunes only", "Sub-Neptunes and super-Earths"),
                  true_values=True, sil_range=(1e-3, 12.0))
    ax_mr.set_xlim(0, 12)
    _draw_density_1x2(ax_hist, arr, cut, nasa, m_sil, r_sil, rng, tag=f"[1x2] {cut_label}")
    # Pared-down styling for this figure: no grid, no VOLATILE/ROCKY labels, and
    # no sample count on NASA.
    ax_mr.set_xlabel(r"Planet mass [$M_\oplus$]")
    ax_mr.set_ylabel(r"Planet radius [$R_\oplus$]")
    for ax in axes:
        ax.grid(False)
        ax.xaxis.label.set_size(24)
        ax.yaxis.label.set_size(24)
        ax.tick_params(labelsize=20)
    for text in [t for t in ax_mr.texts if t.get_text() in ("VOLATILE (above)", "ROCKY (below)")]:
        text.remove()
    h, lbl = ax_mr.get_legend_handles_labels()
    relabel = {"silicate line": r"Pure MgSiO$_3$ mass-radius relation"}
    ax_mr.legend(h, ["Measured exoplanets" if x.startswith("NASA") else relabel.get(x, x)
                     for x in lbl],
                 fontsize=18, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    # Figure-wide title over both panels (placed above the axes; the tight
    # bounding box on save keeps it).
    fig.suptitle(r"Simulated Planet Detections at $I < 50\,I_\oplus$ and $M > 2\,M_\oplus$",
                 fontsize=26, y=1.01, va="bottom")
    fname = "flat_otegi_1x2_cold_cut.png"
    out_png = os.path.join(OUT_DIR, fname)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_DIR / fname, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out_png}")
    print(f"--> Saved paper copy: {PAPER_FIG_DIR / fname}")


def make_paper_2col(pools, nasa, m_sil, r_sil, rng):
    """Paper figure (fig:mrrel): cold super-Earth cut, two representative rocky
    mass-radius relations (Chen & Kipping, Otegi) side by side, mass-radius draw
    on top and volatile-fraction histograms below."""
    print("\n--> Paper 2-col (Chen & Kipping + Otegi, cold super-Earth cut):")
    cut = dict(mass_min=2.0, insol_max=50.0)
    wanted = ["Chen & Kipping 2017", "Otegi et al. 2020"]
    sel = [p for p in pools if p[0] in wanted]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11))
    for ci, (name, eq, applies, arr) in enumerate(sel):
        _draw_scatter(axes[0, ci], arr, cut, nasa, m_sil, r_sil, rng, f"{name}\n{eq}")
        _draw_bells(axes[1, ci], arr, cut, nasa, m_sil, r_sil, rng, tag=f"[2col] {name}")
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "flat_rocky_mr_2col_chen_otegi_cold.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_DIR / "flat_rocky_mr_2col_chen_otegi_cold.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out_png}")
    print(f"--> Saved paper copy: {PAPER_FIG_DIR / 'flat_rocky_mr_2col_chen_otegi_cold.png'}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = puffy_cuts.load_silicate()
    rng = np.random.default_rng(SEED)
    nasa = puffy_cuts.load_nasa()

    print(f"--> building {len(RELATIONS)} rocky-relation pools + detectors "
          f"(mass scatter = {MR_SCATTER_DEX} dex)...")
    pools = [(name, eq, applies, build_arrays(kw, m_sil, r_sil)) for name, eq, applies, kw in RELATIONS]

    for cut_label, cut, fname in CUTS:
        make_figure(cut_label, cut, fname, pools, nasa, m_sil, r_sil, rng)

    otegi_arr = next(arr for name, eq, applies, arr in pools if "Otegi" in name)
    make_otegi_2x2(otegi_arr, nasa, m_sil, r_sil, rng)
    make_otegi_2x1(otegi_arr, nasa, m_sil, r_sil, rng)
    make_paper_2col(pools, nasa, m_sil, r_sil, rng)
    make_otegi_1x2(nasa, m_sil, r_sil, rng)


if __name__ == "__main__":
    main()
DOWNLOAD_NASA_DATA = False  # Set to True to download fresh data, False to use local CSV

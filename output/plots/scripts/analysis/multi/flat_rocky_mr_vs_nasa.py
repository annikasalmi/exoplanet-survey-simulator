"""
flat_rocky_mr_vs_nasa.py — which published ROCKY mass-radius relation, imposed on the
uniform-parameter universe, best reproduces NASA's puffy fraction?

2x4 grid.  Columns = four rocky/small-planet R = C·M^β relations (post-2016; Bashi 2017 excluded
because it has no distinct rocky branch — Chen's Terran branch used instead):
    Chen & Kipping 2017  ·  Otegi et al. 2020  ·  Edmondson et al. 2023  ·  Müller et al. 2024
Each is used as a deterministic "mean MR" (mass from radius) + a common log-normal mass scatter, so the
only thing that changes across columns is the relation itself.

Rows (both show flat A AND flat B):
    1. mass-radius scatter — flat A points + the planets flat A drops (rocky, true M>2, "B only") + NASA.
    2. puffy-fraction bells — flat A vs flat B vs NASA.
flat A = drops rocky planets with TRUE mass > 2 M⊕ ("no rocky super-Earths"); flat B = keeps all.
Detection + measurement noise + the NASA method (precision cut + per-planet perturbation) come from
script 72.

Outputs: one 2x4 per cut (all | insol<50 | Cold Rocky Desert), plus the paper's 2x2
(Otegi only, before/after the cold super-Earth cut) saved to paper/figures.

Run:
    python important_plots/flat_rocky_mr_vs_nasa.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import importlib.util
from pathlib import Path

from tools.paths import LIFESIM_OUTER_DIR, ANALYSIS_DIR, PAPER_FIGURES_DIR, KEPLER_REF_CURVE
ROOT = Path(LIFESIM_OUTER_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import truncnorm
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S72 = _load("s72", str(ROOT / "output" / "plots" / "scripts" / "analysis" / "multi" / "puffy_cuts_flat.py"))
S72.N_REPEATS = 4000

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
SUB_NEPTUNE_FRAC_SD = 0.20       # fractional radius width of the sub-Neptune normal
SUPER_EARTH_FRAC_SD = 0.05       # ... and of each super-Earth normal


def load_earthlike_curve():
    """Earth-like composition curve (ref.ddat: 32% Fe core), sorted by mass."""
    ref = np.loadtxt(KEPLER_REF_CURVE, comments="#")
    ref = ref[ref[:, 0] > 0]
    o = np.argsort(ref[:, 0])
    return ref[o, 0], ref[o, 1]


def otegi_volatile_radius(mass):
    return OTEGI_VOLATILE["mr_C"] * mass ** OTEGI_VOLATILE["mr_beta"]


def build_arrays(rel_kw, m_sil, r_sil, two_populations=False, rng=None):
    """Detected flat-universe pool for one mass-radius relation.

    With two_populations, every planet keeps its mass but gets a new radius:
      * sub-Neptunes (all planets except the rocky super-Earths): a normal
        around Otegi's volatile-rich relation, truncated to stay above the
        silicate line;
      * rocky super-Earths (true rocky, true M > 2 under the base relation; the
        planets only universe B keeps): half from a normal around the same
        volatile-rich relation, half from a normal around the Earth-like
        curve, with no cut.
    Widths (fractional): SUB_NEPTUNE_FRAC_SD for the sub-Neptunes (wide enough
    that some reach below 2.2 R_earth at 8-10 M_earth) and SUPER_EARTH_FRAC_SD
    for each super-Earth normal. Redrawn planets outside the
    radius box (0.5-2.2 R_earth, as for NASA) are dropped. The returned `puffy`
    flags the sub-Neptunes, so the A/B split is by population, not by where a
    planet landed. Detection runs on the redrawn radii.
    """
    cat = generate_flat_catalog(FLAT_N, seed=SEED, mass_model="powerlaw",
                                mass_scatter_dex=MR_SCATTER_DEX, **rel_kw)
    r = pd.to_numeric(cat["radius_p"], errors="coerce")
    m = pd.to_numeric(cat["mass_p"], errors="coerce")
    f = pd.to_numeric(cat["flux_p"], errors="coerce")
    keep = (r.between(S72.BOX["r_lo"], S72.BOX["r_hi"]) & m.between(S72.BOX["m_lo"], S72.BOX["m_hi"])
            & (f.isna() | f.between(S72.BOX["f_lo"], S72.BOX["f_hi"])))
    cat = cat[keep].copy()
    mass = pd.to_numeric(cat["mass_p"], errors="coerce").to_numpy(float)
    radius = pd.to_numeric(cat["radius_p"], errors="coerce").to_numpy(float).copy()
    flux = pd.to_numeric(cat["flux_p"], errors="coerce").to_numpy(float)
    puffy = radius > np.interp(mass, m_sil, r_sil)
    if two_populations:
        se = (~puffy) & (mass > S72.MASS_THRESHOLD)
        sn = ~se
        # Sub-Neptunes: normal around the volatile-rich curve, kept above the silicate line.
        mu = otegi_volatile_radius(mass[sn])
        sd = SUB_NEPTUNE_FRAC_SD * mu
        a = (np.interp(mass[sn], m_sil, r_sil) - mu) / sd
        radius[sn] = truncnorm.rvs(a, np.inf, loc=mu, scale=sd, random_state=rng)
        # Super-Earths: 50/50 mix of the volatile-rich and Earth-like normals, no cut.
        m_se = mass[se]
        centre = np.where(rng.random(m_se.size) < 0.5, otegi_volatile_radius(m_se),
                          np.interp(m_se, *load_earthlike_curve()))
        radius[se] = centre * (1.0 + SUPER_EARTH_FRAC_SD * rng.standard_normal(m_se.size))
        in_box = (radius >= S72.BOX["r_lo"]) & (radius <= S72.BOX["r_hi"])
        cat["radius_p"] = radius
        cat = cat[in_box].copy()
        mass, radius, flux, puffy = mass[in_box], radius[in_box], flux[in_box], sn[in_box]
    td = run_kepler(cat)["detected"].to_numpy(bool)
    rd = run_rv_best(cat, mag_target=S72.RV_MAG_TARGET)["detected"].to_numpy(bool)
    return mass, radius, flux, puffy, td & rd


def noised_scatter_AB(arrays, cut, rng, n_plot=400):
    """One detected + noised draw (with the cut applied); split into flat-A-kept vs dropped-by-A."""
    mass, radius, flux, puffy, det = arrays
    keep = det.copy()
    if cut.get("insol_max"):
        keep = keep & (flux < cut["insol_max"])
    idx = np.flatnonzero(keep)
    mo = mass[idx] * np.exp(rng.normal(0, S72.MASS_FRAC_ERR, idx.size))
    ro = radius[idx] * np.exp(rng.normal(0, S72.RAD_FRAC_ERR, idx.size))
    tmass, tpuffy = mass[idx], puffy[idx]
    if cut.get("mass_min"):
        k = mo > cut["mass_min"]
        mo, ro, tmass, tpuffy = mo[k], ro[k], tmass[k], tpuffy[k]
    dropped = (~tpuffy) & (tmass > S72.MASS_THRESHOLD)          # A drops these (true rocky, true M>2)
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


def true_scatter_AB(arrays, cut, n_plot, rng):
    """Detected planets at their TRUE masses and radii, with the cut applied to
    true values; split into flat-A-kept vs dropped-by-A. Sub-Neptunes on or
    below the silicate line are removed, as in the volatile-fraction MC."""
    mass, radius, flux, puffy, det = arrays
    keep = det.copy()
    if cut.get("insol_max"):
        keep &= flux < cut["insol_max"]
    if cut.get("mass_min"):
        keep &= mass > cut["mass_min"]
    dropped = (~puffy) & (mass > S72.MASS_THRESHOLD)
    keep &= dropped | (radius > np.interp(mass, *S72.load_silicate()))
    idx = np.flatnonzero(keep)
    if idx.size > n_plot:
        idx = rng.choice(idx, n_plot, replace=False)
    return mass[idx], radius[idx], dropped[idx]


def _frac_err_bars(x, frac_sd):
    """1-sigma bars for a log-normal fractional error: [lower, upper] offsets."""
    return np.array([x * (1 - np.exp(-frac_sd)), x * (np.exp(frac_sd) - 1)])


SCATTER_LABELS = ("Escape-only (kept)", "Primordial-rocky: rocky super-Earths (M>2)")


def _draw_scatter(ax, arr, cut, nasa, m_sil, r_sil, rng, title,
                  labels=SCATTER_LABELS, true_values=False):
    """Top-row panel: one detected, noise-perturbed mass-radius draw. With
    true_values, the planets are drawn at their true masses and radii with the
    simulated measurement errors as error bars instead."""
    nmc, nrc, nme1, nme2, nre1, nre2 = nasa_cut(nasa, cut)
    ax.fill_between(m_sil, r_sil, 2.6, color="0.965", zorder=0)
    ax.plot(m_sil, r_sil, "k-", lw=1.2, zorder=6, label="silicate line")
    if true_values:
        mt, rt, dropped = true_scatter_AB(arr, cut, 150, rng)
        for sel, colour, lbl, z in [(~dropped, "tab:blue", labels[0], 3),
                                    (dropped, "tab:orange", labels[1], 4)]:
            ax.errorbar(mt[sel], rt[sel],
                        xerr=_frac_err_bars(mt[sel], S72.MASS_FRAC_ERR),
                        yerr=_frac_err_bars(rt[sel], S72.RAD_FRAC_ERR),
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


def mc_universe_true_sub_neptunes(arrays, drop, cut, m_sil, r_sil, rng):
    """S72.mc_universe on a pool without the planets that are truly on or below the
    silicate line but are not rocky super-Earths (the rocky planets under 2
    M_earth), so the non-super-Earth population is genuinely all sub-Neptunes.
    Measurement noise can still scatter some of them below the line."""
    mass, radius, flux, puffy, det = arrays
    keep = puffy | (mass > S72.MASS_THRESHOLD)
    return S72.mc_universe(tuple(a[keep] for a in arrays), drop, cut, m_sil, r_sil, rng)


BELL_STYLE = (("Escape-only", "tab:orange"), ("Primordial-rocky", "tab:blue"))


def _draw_bells(ax, arr, cut, nasa, m_sil, r_sil, rng, tag="", style=BELL_STYLE,
                mc=None):
    """Bottom-row panel: COUNT histograms of the volatile fraction over the MC draws.
    y = number of the N_REPEATS draws that landed in each bin; N_p = mean planets/draw.
    style gives (label, colour) for universe A then universe B."""
    mc = mc or S72.mc_universe
    nv, n_nasa = S72.mc_nasa(nasa, cut, m_sil, r_sil, rng)
    n_mu, n_sd = nv.mean(), nv.std()
    sA, nA = mc(arr, True, cut, m_sil, r_sil, rng)
    sB, nB = mc(arr, False, cut, m_sil, r_sil, rng)
    all_bell = [b for b in (nv, sA, sB) if b.size]
    cat = np.concatenate(all_bell)
    lo, hi = cat.min(), cat.max(); pad = 0.05 * (hi - lo)
    gx = np.linspace(lo - pad, hi + pad, 400)
    # fixed 0.04-wide bins on the k/25 grid, identical across all figures: NASA's
    # fraction is discrete (same ~25 planets each draw), finer data-driven bins alias it
    edges = np.arange(-0.02, 1.02 + 1e-9, 0.04)
    bw = edges[1] - edges[0]
    y_max = max(np.histogram(b, bins=edges)[0].max() for b in all_bell)
    for (lbl, colour), s, ne in zip(style, (sA, sB), (nA, nB)):
        if s.size == 0:
            continue
        tens = abs(s.mean() - n_mu) / np.sqrt(s.std() ** 2 + n_sd ** 2)
        ax.hist(s, bins=edges, color=colour, alpha=0.30)
        ax.plot(gx, S72.gauss(gx, s.mean(), s.std()) * s.size * bw, color=colour, lw=2.0,
                label=f"{lbl}: $\\mu$={s.mean():.2f} $\\sigma$={s.std():.3f} "
                      f"({tens:.1f}$\\sigma$), N$_p$={ne:.0f}")
        if tag:
            print(f"    {tag} {lbl}: mu={s.mean():.3f} sd={s.std():.3f} "
                  f"tension={tens:.1f}sigma N_planets={ne:.0f} N_draws={s.size}")
    ax.hist(nv, bins=edges, color="tab:green", alpha=0.34)
    ax.plot(gx, S72.gauss(gx, n_mu, n_sd) * nv.size * bw, color="tab:green", lw=2.4,
            label=f"NASA: $\\mu$={n_mu:.2f} $\\sigma$={n_sd:.3f}, N$_p$={n_nasa}")
    ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(0, y_max * 1.15)
    ax.grid(alpha=0.2); ax.legend(fontsize=9, loc="upper left")
    ax.set_xlabel("volatile fraction")
    ax.set_ylabel(f"number of MC draws (of {S72.N_REPEATS:,})")


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


def make_otegi_1x2(nasa, m_sil, r_sil, rng):
    """Side-by-side panel pair: the Otegi mass-radius draw (left) and its
    volatile-fraction count histograms (right), under the cold super-Earth cut.

    Blue = "Sub-Neptunes only" (universe A), orange = "Sub-Neptunes and
    super-Earths" (universe B) in both panels; radii come from build_arrays'
    two-population redraw. The left panel shows true masses and radii with the
    simulated errors as bars; the histograms use the noise-perturbed
    ("measured") values, so noise can put some sub-Neptunes below the line.
    """
    print("\n--> Otegi 1x2 (cold super-Earth cut; two-population radii):")
    cut_label, cut = OTEGI_2X2_CUTS[1]
    otegi_kw = next(kw for name, eq, applies, kw in RELATIONS if "Otegi" in name)
    arr = build_arrays(otegi_kw, m_sil, r_sil, two_populations=True,
                       rng=np.random.default_rng(SEED + 1))
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    _draw_scatter(axes[0], arr, cut, nasa, m_sil, r_sil, rng, "",
                  labels=("Sub-Neptunes only", "Sub-Neptunes and super-Earths"),
                  true_values=True)
    m_line = np.linspace(0.2, 12, 300)
    axes[0].plot(m_line, otegi_volatile_radius(m_line), color="tab:blue", lw=1.2, ls="--",
                 zorder=6, label=r"Otegi volatile-rich ($R=0.70\,M^{0.63}$)")
    axes[0].plot(*load_earthlike_curve(), color="0.5", lw=1.4, zorder=6,
                 label="Earth-like rocky curve")
    axes[0].legend(fontsize=9, loc="lower right", framealpha=0.9)
    axes[0].set_xlim(0, 12)
    _draw_bells(axes[1], arr, cut, nasa, m_sil, r_sil, rng, tag=f"[1x2] {cut_label}",
                style=(("Sub-Neptunes only", "tab:blue"),
                       ("Sub-Neptunes and super-Earths", "tab:orange")),
                mc=mc_universe_true_sub_neptunes)
    fig.tight_layout()
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
    m_sil, r_sil = S72.load_silicate()
    rng = np.random.default_rng(SEED)
    nasa = S72.load_nasa()

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

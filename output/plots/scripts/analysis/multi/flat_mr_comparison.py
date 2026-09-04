"""
19_flat_mr_comparison.py — does a FLAT universe carrying the Chen2017/Forecaster
mass-radius relation reproduce NASA's puffy fraction, and does NASA look more like "rocky
super-Earths exist" (universe B) or "they don't" (universe A)?

3x3 grid.  Columns = three cuts:  all (no cut) | insolation < 50 I⊕ | mass>2 M⊕ & insol<50
(the cold super-Earth corner).  Rows:
  1. mass-radius scatter of the flat (Forecaster-MR) population + NASA, with the silicate curve.
  2. puffy-fraction bells: flat A vs flat B vs NASA   (flat universe, Forecaster MR).
  3. puffy-fraction bells: P-Pop A vs P-Pop B vs NASA  (real P-Pop universe, for reference).

flat/P-Pop A = drops rocky planets with TRUE mass > 2 M⊕ ("no rocky super-Earths"); B = keeps all.
Both are transit+RV detected and measurement-noised the same way as script 72 (which supplies the
machinery + the updated NASA method: precision cut 25%/8% + per-planet perturbation, no bootstrap).

The flat universe here uses mass_model="ppop" (P-Pop's Forecaster R->M), so its puffy fraction is a
fair, physical comparison to NASA. The heavy Forecaster mass assignment is cached to disk.

Run:
    python scripts/19_flat_mr_comparison.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from run.ppop.uniform_generator import get_or_build_catalog, UNIFORM_OUT_DIR
from run.ppop.flat_detect import run_kepler, run_rv_best


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# reuse script 72's puffy machinery + updated NASA method (precision cut + per-planet perturbation)
S72 = _load("s72", str(ROOT / "output" / "plots" / "scripts" / "analysis" / "multi" / "puffy_cuts_flat.py"))
S72.N_REPEATS = 4000            # bells smooth at 4k; keeps the 12-panel run quick

OUT_DIR = os.path.join(ROOT, "output/plots", "19_flat_mr_comparison")
FLAT_N = 200000                 # flat Forecaster pool (cached; first build is the slow part)
SEED = 0
FLAT_CACHE = os.path.join(UNIFORM_OUT_DIR, f"flat_catalog_forecaster_n{FLAT_N}_s{SEED}.csv")

CUTS = [("all (no cut)", {}),
        ("insolation < 50 I⊕", dict(insol_max=50.0)),
        ("mass>2 & insol<50", dict(mass_min=2.0, insol_max=50.0))]
# (drop, label, colour) — A drops rocky true-mass>2; B keeps all
FLAT_MODELS = [(True, "flat A", "tab:orange"), (False, "flat B", "tab:blue")]
PPOP_MODELS = [(True, "P-Pop A", "tab:red"), (False, "P-Pop B", "tab:purple")]


def flat_arrays(m_sil, r_sil):
    """Flat universe with Forecaster (P-Pop) mass-radius relation; same array format as S72.build_pool."""
    cat = get_or_build_catalog(FLAT_CACHE, mass_model="ppop", n_planets=FLAT_N, seed=SEED)
    import pandas as pd
    r = pd.to_numeric(cat["radius_p"], errors="coerce")
    m = pd.to_numeric(cat["mass_p"], errors="coerce")
    f = pd.to_numeric(cat["flux_p"], errors="coerce")
    keep = (r.between(S72.BOX["r_lo"], S72.BOX["r_hi"]) & m.between(S72.BOX["m_lo"], S72.BOX["m_hi"])
            & (f.isna() | f.between(S72.BOX["f_lo"], S72.BOX["f_hi"])))
    cat = cat[keep].copy()
    mass = pd.to_numeric(cat["mass_p"], errors="coerce").to_numpy(float)
    radius = pd.to_numeric(cat["radius_p"], errors="coerce").to_numpy(float)
    flux = pd.to_numeric(cat["flux_p"], errors="coerce").to_numpy(float)
    puffy = radius > np.interp(mass, m_sil, r_sil)
    td = run_kepler(cat)["detected"].to_numpy(bool)
    rd = run_rv_best(cat, mag_target=S72.RV_MAG_TARGET)["detected"].to_numpy(bool)
    return mass, radius, flux, puffy, td & rd


def noised_scatter(arrays, cut, rng, n_plot=350):
    mass, radius, flux, puffy, det = arrays
    keep = np.ones(len(mass), bool)
    if cut.get("insol_max"):
        keep = keep & (flux < cut["insol_max"])
    idx = np.flatnonzero(keep & det)
    mo = mass[idx] * np.exp(rng.normal(0, S72.MASS_FRAC_ERR, idx.size))
    ro = radius[idx] * np.exp(rng.normal(0, S72.RAD_FRAC_ERR, idx.size))
    if cut.get("mass_min"):
        k = mo > cut["mass_min"]; mo, ro = mo[k], ro[k]
    if mo.size > n_plot:
        j = rng.choice(mo.size, n_plot, replace=False); mo, ro = mo[j], ro[j]
    return mo, ro


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = S72.load_silicate()
    rng = np.random.default_rng(SEED)

    print("--> building flat (Forecaster MR) + P-Pop pools + detectors...")
    flat = flat_arrays(m_sil, r_sil)
    ppop = S72.build_pool("ppop", m_sil, r_sil)
    nasa = S72.load_nasa()

    # ---- pass 1: compute every panel ----
    panels = {}            # (row, col) -> dict
    all_bell = []
    print(f"\n  {'row':<6}{'cut':<20}{'model':<9}{'μ':>7}{'σ':>7}{'tension':>9}{'N_eff':>7}")
    for ci, (cut_label, cut) in enumerate(CUTS):
        nv, n_nasa = S72.mc_nasa(nasa, cut, m_sil, r_sil, rng)
        n_mu, n_sd = nv.mean(), nv.std()
        all_bell.append(nv)
        for ri, (arrays, models, row) in enumerate([(flat, FLAT_MODELS, "flat"),
                                                     (ppop, PPOP_MODELS, "P-Pop")]):
            series = []
            for drop, label, colour in models:
                s, neff = S72.mc_universe(arrays, drop, cut, m_sil, r_sil, rng)
                tens = (abs(s.mean() - n_mu) / np.sqrt(s.std() ** 2 + n_sd ** 2)
                        if s.size and (s.std() + n_sd) > 0 else np.nan)
                series.append((label, colour, s, neff, tens))
                if s.size:
                    all_bell.append(s)
                    print(f"  {row:<6}{cut_label:<20}{label:<9}{s.mean():>7.3f}{s.std():>7.3f}"
                          f"{tens:>8.1f}σ{neff:>7.0f}")
            panels[(ri, ci)] = dict(series=series, nv=nv, n_nasa=n_nasa, n_mu=n_mu, n_sd=n_sd)
        print()

    # shared bell axes
    cat_bell = np.concatenate(all_bell)
    lo, hi = cat_bell.min(), cat_bell.max(); pad = 0.05 * (hi - lo)
    gx = np.linspace(lo - pad, hi + pad, 400)
    edges = np.linspace(lo - pad, hi + pad, 55)
    y_max = 0.0
    for P in panels.values():
        for _, _, s, _, _ in P["series"]:
            if s.size:
                y_max = max(y_max, np.histogram(s, bins=edges, density=True)[0].max())
        y_max = max(y_max, np.histogram(P["nv"], bins=edges, density=True)[0].max())

    # ---- pass 2: plot 3x3 ----
    fig, axes = plt.subplots(3, 3, figsize=(19, 15))
    for ci, (cut_label, cut) in enumerate(CUTS):
        # row 0: M-R scatter of the flat (Forecaster MR) population + NASA
        ax = axes[0, ci]
        ax.fill_between(m_sil, r_sil, 2.6, color="0.965", zorder=0)
        ax.plot(m_sil, r_sil, "k-", lw=1.2, zorder=6, label="silicate rock limit")
        mo, ro = noised_scatter(flat, cut, rng)
        ax.scatter(mo, ro, s=16, color="tab:blue", alpha=0.45, lw=0, zorder=3, label="flat (Forecaster MR)")
        nmc, nrc = nasa["m"], nasa["r"]
        if cut.get("insol_max"):
            sel = nasa["ins"] < cut["insol_max"]; nmc, nrc = nmc[sel], nrc[sel]
        if cut.get("mass_min"):
            sel = nmc > cut["mass_min"]; nmc, nrc = nmc[sel], nrc[sel]
        ax.scatter(nmc, nrc, s=30, facecolor="none", edgecolor="k", lw=1.0, zorder=5,
                   label=f"NASA (N={nmc.size})")
        ax.set_xlim(0, 13); ax.set_ylim(0.5, 2.4)
        ax.set_title(f"{cut_label}\nmass-radius scatter (flat, Forecaster MR)", fontsize=10)
        ax.grid(alpha=0.2); ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
        ax.text(0.2, 2.3, "PUFFY (above)", fontsize=8, color="0.4", va="top")
        ax.text(4.0, 0.6, "ROCKY (below)", fontsize=8, color="0.4")
        if ci == 0:
            ax.set_ylabel(r"planet radius [$R_\oplus$]")
        ax.set_xlabel(r"planet mass [$M_\oplus$]")

        # rows 1-2: puffy-fraction bells (flat A/B, then P-Pop A/B) vs NASA
        for ri, row_label in [(0, "flat (Forecaster MR)"), (1, "P-Pop")]:
            ax = axes[ri + 1, ci]
            P = panels[(ri, ci)]
            for label, colour, s, neff, tens in P["series"]:
                if s.size == 0:
                    continue
                ax.hist(s, bins=edges, density=True, color=colour, alpha=0.30)
                ax.plot(gx, S72.gauss(gx, s.mean(), s.std()), color=colour, lw=2.0,
                        label=f"{label}: μ={s.mean():.2f} σ={s.std():.3f} ({tens:.1f}σ)")
            ax.hist(P["nv"], bins=edges, density=True, color="tab:green", alpha=0.34)
            ax.plot(gx, S72.gauss(gx, P["n_mu"], P["n_sd"]), color="tab:green", lw=2.4,
                    label=f"NASA: μ={P['n_mu']:.2f} σ={P['n_sd']:.3f} (N={P['n_nasa']})")
            ax.set_xlim(gx[0], gx[-1]); ax.set_ylim(0, y_max * 1.08)
            ax.set_title(f"{row_label} — puffy fraction", fontsize=10)
            ax.grid(alpha=0.2); ax.legend(fontsize=7.6, loc="upper left")
            ax.set_xlabel("puffy fraction")
            if ci == 0:
                ax.set_ylabel("density over repeated draws")

    fig.suptitle("Uniform-parameter universe with the Chen2017/Forecaster mass-radius relation vs NASA, under three cuts\n"
                 "row 1: M-R scatter · row 2: flat A/B puffy bells · row 3: P-Pop A/B (reference); "
                 "A = no rocky super-Earths (drop rocky M>2), B = keep all; closest bell to green NASA wins",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = os.path.join(OUT_DIR, "flat_forecaster_3x3_cuts.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out_png}")


if __name__ == "__main__":
    main()

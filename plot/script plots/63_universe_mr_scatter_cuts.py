"""
63_universe_mr_scatter_cuts.py — SEE the sampled planet populations on the mass-radius diagram
(with the silicate rocky/puffy curve), for four sub-samples, so the puffy fraction becomes visual.

These are exactly the planets that go INTO the puffy-fraction distributions: one representative
detected + measurement-noised draw of each universe, plus the real NASA planets. A planet ABOVE
the black silicate curve is "puffy", below it is "rocky" -- so the puffy fraction is just the
fraction of points above the curve.

Panels (1x4):  all (no cut) | mass > 2 M_earth | insolation < 50 I_earth | mass>2 & insol<50.

Also answers: why are flat A / P-Pop A (which have NO rocky planet with true mass > threshold)
not at puffy fraction = 1 after the mass cut? -> printed decomposition (it is measurement-noise
leakage, not real rocky planets above the threshold).

Run:
    python scripts/63_universe_mr_scatter_cuts.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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

from run.ppop.uniform_generator import generate_flat_catalog
from run.ppop.flat_detect import run_kepler, run_rv_best

SILICATE_CURVE = ROOT / "Hongyi-silicon.ddat"
PPOP_CATALOG = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = os.path.join(ROOT, "my_outputs", "63_universe_mr_scatter_cuts")

MASS_THRESHOLD = 2.0
MASS_FRAC_ERR = 0.20
RAD_FRAC_ERR = 0.046
NASA_MASS_PREC = 0.25           # keep NASA planets with fractional mass error <= this
NASA_RAD_PREC = 0.08            # keep NASA planets with fractional radius error <= this
FLAT_N_POOL = 300000
RNG_SEED = 1
RV_MAG_TARGET = 12.0
N_PLOT = 250                    # points plotted per universe (subsample for clarity)
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)

# cosmetics
PUFFY_BG = "0.965"              # dimmer puffy-region shading (was 0.92)
DOT_SIZE = 16                   # fake-planet marker size (was 7; still < NASA 30)
DOT_ALPHA = 0.45
SIL_LW = 1.2                    # silicate curve linewidth (was 2.5)

# two separate figures: flat universes and P-Pop universes
FLAT_UNIS = [("flat", True,  "flat A", "tab:orange"),
             ("flat", False, "flat B", "dodgerblue")]
PPOP_UNIS = [("ppop", True,  "P-Pop A", "crimson"),
             ("ppop", False, "P-Pop B", "purple")]
GROUPS = [("uniform-parameter universe", FLAT_UNIS, "flat_mr_scatter_cuts.png"),
          ("P-Pop universe", PPOP_UNIS, "ppop_mr_scatter_cuts.png")]
UNIVERSES = FLAT_UNIS + PPOP_UNIS   # for the printed A-universe diagnostic
CUTS = [
    ("all (no cut)", {}),
    ("mass > 2 M⊕", dict(mass_min=2.0)),
    ("insolation < 50 I⊕", dict(insol_max=50.0)),
    ("mass>2 & insol<50", dict(mass_min=2.0, insol_max=50.0)),
]


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def build_pool(population, m_sil, r_sil):
    if population == "flat":
        pool = generate_flat_catalog(n_planets=FLAT_N_POOL, seed=0)
    else:
        cols = ["radius_p", "mass_p", "p_orb", "inc_p", "ecc_p", "semimajor_p", "radius_s",
                "mass_s", "temp_s", "teff_s", "distance_s", "l_sun", "flux_p", "detected"]
        pool = pd.read_csv(PPOP_CATALOG, usecols=lambda c: c in cols, low_memory=False)
    r = pd.to_numeric(pool["radius_p"], errors="coerce")
    m = pd.to_numeric(pool["mass_p"], errors="coerce")
    keep = r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    f = pd.to_numeric(pool["flux_p"], errors="coerce")
    keep = keep & (f.isna() | f.between(BOX["f_lo"], BOX["f_hi"]))
    pool = pool[keep].copy()
    mass = pd.to_numeric(pool["mass_p"], errors="coerce").to_numpy(float)
    radius = pd.to_numeric(pool["radius_p"], errors="coerce").to_numpy(float)
    flux = pd.to_numeric(pool["flux_p"], errors="coerce").to_numpy(float)
    puffy = radius > np.interp(mass, m_sil, r_sil)
    if population == "flat":
        td = run_kepler(pool)["detected"].to_numpy(bool)
    else:
        td = pd.to_numeric(pool["detected"], errors="coerce").fillna(0).astype(bool).to_numpy()
    rd = run_rv_best(pool, mag_target=RV_MAG_TARGET)["detected"].to_numpy(bool)
    return mass, radius, flux, puffy, td & rd


def model_observed(arrays, drop, cut, rng):
    """One detected + noised draw of the universe with the cut applied. Returns observed M,R and true mask info."""
    mass, radius, flux, puffy, det = arrays
    keep = ~((~puffy) & (mass > MASS_THRESHOLD)) if drop else np.ones(len(mass), bool)
    if cut.get("insol_max"):
        keep = keep & (flux < cut["insol_max"])
    idx = np.flatnonzero(keep & det)
    mo = mass[idx] * np.exp(rng.normal(0, MASS_FRAC_ERR, idx.size))
    ro = radius[idx] * np.exp(rng.normal(0, RAD_FRAC_ERR, idx.size))
    true_mass = mass[idx]; true_puffy = puffy[idx]
    if cut.get("mass_min"):
        k = mo > cut["mass_min"]
        mo, ro, true_mass, true_puffy = mo[k], ro[k], true_mass[k], true_puffy[k]
    return mo, ro, true_mass, true_puffy


def puffy_fraction(mo, ro, m_sil, r_sil):
    return float((ro > np.interp(mo, m_sil, r_sil)).mean()) if mo.size else np.nan


def load_nasa():
    df = pd.read_csv(NASA_FILE)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    ins = pd.to_numeric(df["pl_insol"], errors="coerce")
    me = np.maximum(pd.to_numeric(df["pl_bmasseerr1"], errors="coerce").abs(),
                    pd.to_numeric(df["pl_bmasseerr2"], errors="coerce").abs())
    re = np.maximum(pd.to_numeric(df["pl_radeerr1"], errors="coerce").abs(),
                    pd.to_numeric(df["pl_radeerr2"], errors="coerce").abs())
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    meas = prov.str.contains("Mass|Msini", case=False, na=False) & ~prov.str.contains("Calc", case=False, na=False)
    prec = (me / m <= NASA_MASS_PREC) & (re / r <= NASA_RAD_PREC)   # measurement-precision test
    keep = meas & prec & r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    return m[keep].to_numpy(), r[keep].to_numpy(), ins[keep].to_numpy()


def nasa_cut(nm, nr, nins, cut):
    sel = np.ones(len(nm), bool)
    if cut.get("insol_max"):
        sel = sel & (nins < cut["insol_max"])
    if cut.get("mass_min"):
        sel = sel & (nm > cut["mass_min"])
    return nm[sel], nr[sel]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    rng = np.random.default_rng(RNG_SEED)
    print("--> building pools + detectors...")
    pools = {pop: build_pool(pop, m_sil, r_sil) for pop in ["flat", "ppop"]}
    nm, nr, nins = load_nasa()

    # ---- Q1: why are A universes not at puffy fraction 1 after the mass>2 cut? ----
    print("\n  WHY flat A / P-Pop A are not at puffy fraction = 1 after the mass>2 cut")
    print("  (universe A has ZERO rocky planets with TRUE mass > 2 by construction):")
    for pop, _, label, _ in [u for u in UNIVERSES if u[1]]:   # the A universes
        mo, ro, tmass, tpuffy = model_observed(pools[pop], True, dict(mass_min=2.0), np.random.default_rng(7))
        obs_puffy = ro > np.interp(mo, m_sil, r_sil)
        rocky_obs = ~obs_puffy
        leaked = rocky_obs & (tmass <= 2.0)            # true-rocky kept below 2, noise pushed mass>2
        flipped = rocky_obs & tpuffy                   # truly puffy, noise flipped the label
        true_rocky_above = rocky_obs & (~tpuffy) & (tmass > 2.0)  # should be ~0
        print(f"    {label}: observed puffy fraction={obs_puffy.mean():.3f} (noise-free would be 1.000)")
        print(f"      of the {rocky_obs.sum()} 'rocky' points: {leaked.sum()} leaked from TRUE mass<=2, "
              f"{flipped.sum()} are truly-puffy flipped by noise, {true_rocky_above.sum()} real rocky w/ true mass>2")

    # ---- two figures: FLAT and P-Pop, each a 2x2 M-R scatter over the four cuts ----
    for group_label, unis, fname in GROUPS:
        make_figure(group_label, unis, fname, pools, nm, nr, nins, m_sil, r_sil, rng)


def make_figure(group_label, unis, fname, pools, nm, nr, nins, m_sil, r_sil, rng):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    for ax, (cut_label, cut) in zip(axes.flat, CUTS):
        ax.fill_between(m_sil, r_sil, 2.6, color=PUFFY_BG, zorder=0)   # puffy region shading (above)
        ax.plot(m_sil, r_sil, "k-", lw=SIL_LW, zorder=6, label="silicate rock limit")
        title_bits = []
        for pop, drop, label, colour in unis:
            mo, ro, _, _ = model_observed(pools[pop], drop, cut, rng)
            pf = puffy_fraction(mo, ro, m_sil, r_sil)
            title_bits.append(f"{label} {pf:.2f}")
            if mo.size > N_PLOT:
                j = rng.choice(mo.size, N_PLOT, replace=False)
                mo, ro = mo[j], ro[j]
            ax.scatter(mo, ro, s=DOT_SIZE, color=colour, alpha=DOT_ALPHA, lw=0, zorder=3,
                       label=f"{label} (puffy {pf:.2f})")
        nmc, nrc = nasa_cut(nm, nr, nins, cut)
        pf_nasa = puffy_fraction(nmc, nrc, m_sil, r_sil)
        ax.scatter(nmc, nrc, s=30, facecolor="none", edgecolor="k", lw=1.0, zorder=5,
                   label=f"NASA (puffy {pf_nasa:.2f}, N={nmc.size})")
        ax.set_xlim(0, 13); ax.set_ylim(0.5, 2.4)
        ax.set_title(f"{cut_label}\npuffy frac → " + "  ".join(title_bits) + f"  NASA {pf_nasa:.2f}", fontsize=9)
        ax.grid(alpha=0.2, which="both")
        ax.legend(fontsize=7.6, loc="lower right", framealpha=0.9)
        ax.text(0.13, 2.3, "PUFFY (above)", fontsize=8, color="0.4", va="top")
        ax.text(4.0, 0.6, "ROCKY (below)", fontsize=8, color="0.4")
    for ax in axes[-1, :]:
        ax.set_xlabel(r"planet mass [$M_\oplus$]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"planet radius [$R_\oplus$]")

    fig.suptitle(f"{group_label}: sampled population on the mass-radius diagram (transit+RV detected + measurement noise)\n"
                 f"above the black silicate curve = puffy, below = rocky (Earth itself sits below it — iron core); "
                 f"NASA = open circles (measurement-precision test: mass ≤{NASA_MASS_PREC:.0%}, radius ≤{NASA_RAD_PREC:.0%}).",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out_png = os.path.join(OUT_DIR, fname)
    fig.savefig(out_png, dpi=175, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out_png}")


if __name__ == "__main__":
    main()

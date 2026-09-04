"""
27_mstar_conditioned.py — condition the rocky/puffy comparison on HOST-STAR TYPE.

Part 1 showed transit favours big stars and RV favours small stars, and the flat universe
over-weights hot stars. Conditioning on M dwarfs removes that confound. And YES, this works
for the uniform/flat universe too: the flat generator draws Teff ~ U[2300,9700], so we simply
keep the planets whose host has Teff < 3900 K (~16% of the flat pool).

Top row  = ALL host stars.   Bottom row = M-DWARF hosts only (Teff < 3900 K).
Columns  = the three "location" metrics that actually track NASA: puffy fraction,
           mean distance to silicate curve, median bulk density. (Scatter/GMM dropped.)

Run:
    python scripts/27_mstar_conditioned.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
from pathlib import Path

ROOT = Path(LIFESIM_OUTER_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from tools.paths import LIFESIM_OUTER_DIR, SILICON_CURVE
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
from run.ppop import universe_metrics as UM

SILICATE_CURVE = Path(SILICON_CURVE)
PPOP_CATALOG = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = os.path.join(ROOT, "output/plots", "27_mstar_conditioned")

N_SAMPLE = 20000
N_REPEATS = 5000
MASS_THRESHOLD = 2.0
MDWARF_TEFF = 3900.0
MASS_FRAC_ERR, RAD_FRAC_ERR = 0.20, 0.047
FLAT_N_POOL = 300000
RNG_SEED = 0
RV_MAG_TARGET = 12.0
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)

UNIVERSES = [
    ("flat", True,  "flat A", "tab:orange"),
    ("flat", False, "flat B", "tab:blue"),
    ("ppop", True,  "P-Pop A", "tab:red"),
    ("ppop", False, "P-Pop B", "tab:purple"),
]
METRICS = [
    ("puffy", "puffy fraction", "{:.2f}", lambda M, R, ms, rs: UM.puffy_fraction(M, R, ms, rs)),
    ("dist", "mean dist to silicate curve [R⊕]", "{:+.3f}", lambda M, R, ms, rs: UM.mean_distance_to_curve(M, R, ms, rs)),
    ("dens", "median bulk density [g/cc]", "{:.2f}", lambda M, R, ms, rs: UM.median_density(M, R)),
]


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def teff_of(df):
    for c in ["teff_s", "temp_s"]:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().any():
            return pd.to_numeric(df[c], errors="coerce").to_numpy(float)
    return np.full(len(df), np.nan)


def build_pool(population, m_sil, r_sil):
    if population == "flat":
        pool = generate_flat_catalog(n_planets=FLAT_N_POOL, seed=RNG_SEED)
    else:
        pool = pd.read_csv(PPOP_CATALOG, low_memory=False)
    r = pd.to_numeric(pool["radius_p"], errors="coerce")
    m = pd.to_numeric(pool["mass_p"], errors="coerce")
    keep = r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    if "flux_p" in pool.columns:
        f = pd.to_numeric(pool["flux_p"], errors="coerce")
        keep = keep & (f.isna() | f.between(BOX["f_lo"], BOX["f_hi"]))
    pool = pool[keep].copy()
    mass = pd.to_numeric(pool["mass_p"], errors="coerce").to_numpy(float)
    radius = pd.to_numeric(pool["radius_p"], errors="coerce").to_numpy(float)
    teff = teff_of(pool)
    puffy = radius > np.interp(mass, m_sil, r_sil)
    if population == "flat":
        td = run_kepler(pool)["detected"].to_numpy(bool)
    else:
        td = pd.to_numeric(pool["detected"], errors="coerce").fillna(0).astype(bool).to_numpy()
    rd = run_rv_best(pool, mag_target=RV_MAG_TARGET)["detected"].to_numpy(bool)
    return mass, radius, teff, puffy, td & rd


def mc(mass, radius, teff, puffy, det, drop, mdwarf, m_sil, r_sil, rng):
    keep = ~((~puffy) & (mass > MASS_THRESHOLD)) if drop else np.ones(len(mass), bool)
    if mdwarf:
        keep = keep & (teff < MDWARF_TEFF)
    idx = np.flatnonzero(keep)
    if idx.size < 50:
        return {k: np.array([]) for k, *_ in METRICS}, 0.0
    m_u, r_u, det_u = mass[idx], radius[idx], det[idx]
    L = idx.size
    out = {k: np.full(N_REPEATS, np.nan) for k, *_ in METRICS}
    neff = []
    for i in range(N_REPEATS):
        s = rng.integers(0, L, N_SAMPLE)
        sel = s[det_u[s]]
        if sel.size < 20:
            continue
        Mo = m_u[sel] * np.exp(rng.normal(0, MASS_FRAC_ERR, sel.size))
        Ro = r_u[sel] * np.exp(rng.normal(0, RAD_FRAC_ERR, sel.size))
        for k, _, _, fn in METRICS:
            out[k][i] = fn(Mo, Ro, m_sil, r_sil)
        neff.append(sel.size)
    return out, float(np.mean(neff)) if neff else 0.0


def load_nasa():
    df = pd.read_csv(NASA_FILE)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    me1 = pd.to_numeric(df["pl_bmasseerr1"], errors="coerce").abs()
    me2 = pd.to_numeric(df["pl_bmasseerr2"], errors="coerce").abs()
    re1 = pd.to_numeric(df["pl_radeerr1"], errors="coerce").abs()
    re2 = pd.to_numeric(df["pl_radeerr2"], errors="coerce").abs()
    teff = pd.to_numeric(df["st_teff"], errors="coerce")
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    measured = prov.str.contains("Mass|Msini", case=False, na=False) & ~prov.str.contains("Calc", case=False, na=False)
    keep = measured & r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    out = dict(m=m[keep].to_numpy(), r=r[keep].to_numpy(),
               me1=me1[keep].to_numpy(), me2=me2[keep].to_numpy(),
               re1=re1[keep].to_numpy(), re2=re2[keep].to_numpy(),
               teff=teff[keep].to_numpy())
    for k in ["me1", "me2"]:
        out[k] = np.where(np.isfinite(out[k]), out[k], MASS_FRAC_ERR * out["m"])
    for k in ["re1", "re2"]:
        out[k] = np.where(np.isfinite(out[k]), out[k], RAD_FRAC_ERR * out["r"])
    return out


def mc_nasa(d, mdwarf, m_sil, r_sil, rng):
    sub = (d["teff"] < MDWARF_TEFF) if mdwarf else np.ones(len(d["m"]), bool)
    m, r = d["m"][sub], d["r"][sub]
    me1, me2, re1, re2 = d["me1"][sub], d["me2"][sub], d["re1"][sub], d["re2"][sub]
    n = len(m)
    out = {k: np.full(N_REPEATS, np.nan) for k, *_ in METRICS}
    for i in range(N_REPEATS):
        b = rng.integers(0, n, n)
        zm, zr = rng.normal(size=n), rng.normal(size=n)
        mb = np.clip(m[b] + np.where(zm >= 0, zm * me1[b], zm * me2[b]), 1e-3, None)
        rb = np.clip(r[b] + np.where(zr >= 0, zr * re1[b], zr * re2[b]), 1e-3, None)
        for k, _, _, fn in METRICS:
            out[k][i] = fn(mb, rb, m_sil, r_sil)
    return out, n


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) if sd > 0 else np.zeros_like(x)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    rng = np.random.default_rng(RNG_SEED)
    print("--> building pools + detectors...")
    pools = {pop: build_pool(pop, m_sil, r_sil) for pop in ["flat", "ppop"]}
    nasa = load_nasa()

    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    for row, mdwarf in enumerate([False, True]):
        nasa_out, n_nasa = mc_nasa(nasa, mdwarf, m_sil, r_sil, rng)
        results = []
        for pop, drop, label, colour in UNIVERSES:
            mass, radius, teff, puffy, det = pools[pop]
            out, neff = mc(mass, radius, teff, puffy, det, drop, mdwarf, m_sil, r_sil, rng)
            results.append((label, colour, out, neff))
        tag = "M-dwarf hosts only (Teff<3900)" if mdwarf else "ALL host stars"
        print(f"\n  [{tag}]  NASA N={n_nasa}")
        for ci, (key, title, fmt, _) in enumerate(METRICS):
            ax = axes[row, ci]
            nval = nasa_out[key][np.isfinite(nasa_out[key])]
            n_mu, n_sd = nval.mean(), nval.std()
            allv = [nval] + [r[2][key][np.isfinite(r[2][key])] for r in results]
            cat = np.concatenate([a for a in allv if a.size])
            lo, hi = cat.min(), cat.max(); pad = 0.05 * (hi - lo + 1e-9)
            edges = np.linspace(lo - pad, hi + pad, 60); xs = np.linspace(lo - pad, hi + pad, 400)
            for label, colour, out, neff in results:
                s = out[key][np.isfinite(out[key])]
                if s.size == 0:
                    continue
                tens = abs(s.mean() - n_mu) / n_sd if n_sd > 0 else np.nan
                ax.hist(s, bins=edges, density=True, color=colour, alpha=0.25)
                ax.plot(xs, gauss(xs, s.mean(), s.std()), color=colour, lw=1.9,
                        label=f"{label}: μ={fmt.format(s.mean())} ({tens:.1f}σ, N_det≈{neff:.0f})")
                if ci == 0:
                    print(f"    {label:<8} {key}: μ={fmt.format(s.mean())}  tension={tens:.2f}σ  N_det≈{neff:.0f}")
            ax.hist(nval, bins=edges, density=True, color="tab:green", alpha=0.32)
            ax.plot(xs, gauss(xs, n_mu, n_sd), color="tab:green", lw=2.3,
                    label=f"NASA: μ={fmt.format(n_mu)} (N={n_nasa})")
            ax.set_title(f"{title}\n[{tag}]", fontsize=10)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7.6, loc="best")
    axes[0, 0].set_ylabel("density over draws — ALL hosts")
    axes[1, 0].set_ylabel("density over draws — M dwarfs only")
    fig.suptitle("Conditioning on host-star type — does matching stellar samples change which universe fits NASA?\n"
                 "transit+RV detected, measurement-error propagated; M-dwarf row puts all universes on the same star type",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_png = os.path.join(OUT_DIR, "mstar_conditioned.png")
    fig.savefig(out_png, dpi=175, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


if __name__ == "__main__":
    main()

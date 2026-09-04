"""
22_metrics_vs_nasa.py — "which fake universe is closest to the OBSERVED one?"
done honestly, with measurement-error propagation and two complementary metrics.

WHAT THIS FIXES vs script 56
----------------------------
Script 56 showed only sampling (statistical) error and a hard puffy/rocky label from TRUE
mass & radius. Here we forward-model every universe to the OBSERVED plane and propagate
measurement error, so the comparison to NASA is apples-to-apples:

  model universe:  true (M,R)  --transit+RV detector-->  detected  --add measurement noise-->
                   "observed" (M,R)  -->  metric
  NASA:            reported (M,R) with reported asymmetric errors  -->  bootstrap + perturb  -->  metric

NASA's "transiting + measured-mass" planets are themselves selected by transit AND RV/TTV,
so the transit+RV-detected model subset is the right comparison.

Measurement noise (log-normal, from NASA medians):  mass ~20%, radius ~4.7%.

TWO COMPLEMENTARY METRICS (run side by side):
  V1  puffy fraction       N_puffy/(N_puffy+N_rocky)   (uses the silicate curve)
  V2  median bulk density  5.513 * M / R^3  [g/cc]     (CURVE-FREE cross-check)
The pair answers the same question two independent ways -- with the curve (V1) and
without it (V2) -- so the result does not hinge on where the silicate curve is drawn.
(The 1-D "mean distance to curve" was dropped: it is curve-dependent like V1, so it adds
a second curve-based number rather than an independent check.)

"Closest to observed" = smallest tension |mu_model - mu_NASA| / sqrt(sigma_model^2 + sigma_NASA^2).

Run:
    python scripts/22_metrics_vs_nasa.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from tools.paths import SILICON_CURVE
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

SILICATE_CURVE = Path(SILICON_CURVE)
PPOP_CATALOG = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = os.path.join(ROOT, "output/plots", "22_metrics_vs_nasa")

# ============================ KNOBS ============================
N_SAMPLE_PER_UNIVERSE = 20000   # planets grabbed per draw
N_REPEATS = 10000               # draws per universe / NASA bootstraps
MASS_THRESHOLD = 2.0            # universe A drops rocky planets above this (M_earth)
MASS_FRAC_ERR = 0.20            # measurement noise applied to MODEL planets (from NASA median)
RAD_FRAC_ERR = 0.047
# ==============================================================

FLAT_N_POOL = 300000
RNG_SEED = 0
RV_MAG_TARGET = 12.0
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)
RHO_EARTH = 5.513  # g/cc

UNIVERSES = [
    ("flat", True,  f"flat A: no rocky M>{MASS_THRESHOLD:g}", "tab:orange"),
    ("flat", False, "flat B: with rocky",                    "tab:blue"),
    ("ppop", True,  f"P-Pop A: no rocky M>{MASS_THRESHOLD:g}", "tab:red"),
    ("ppop", False, "P-Pop B: with rocky",                   "tab:purple"),
]
METRICS = [
    ("puffy", "puffy fraction  N_puffy/(N_puffy+N_rocky)", "{:.2f}"),
    ("dens",  "median bulk density  5.513 M/R^3  [g/cc]  (curve-free)", "{:.2f}"),
]


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def metrics(m_obs, r_obs, m_sil, r_sil):
    sil = np.interp(m_obs, m_sil, r_sil)
    puffy = float((r_obs > sil).mean())
    dens = float(np.median(RHO_EARTH * m_obs / r_obs ** 3))
    return puffy, dens


def build_pool(population, m_sil, r_sil):
    """true mass, radius, puffy flag, and detected (transit AND RV) per planet."""
    if population == "flat":
        pool = generate_flat_catalog(n_planets=FLAT_N_POOL, seed=RNG_SEED)
    else:
        cols = ["radius_p", "mass_p", "p_orb", "inc_p", "ecc_p", "semimajor_p", "radius_s",
                "mass_s", "temp_s", "teff_s", "distance_s", "l_sun", "flux_p", "detected"]
        pool = pd.read_csv(PPOP_CATALOG, usecols=lambda c: c in cols, low_memory=False)

    r = pd.to_numeric(pool["radius_p"], errors="coerce")
    m = pd.to_numeric(pool["mass_p"], errors="coerce")
    keep = r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    if "flux_p" in pool.columns:
        f = pd.to_numeric(pool["flux_p"], errors="coerce")
        keep = keep & (f.isna() | f.between(BOX["f_lo"], BOX["f_hi"]))
    pool = pool[keep].copy()

    mass = pd.to_numeric(pool["mass_p"], errors="coerce").to_numpy(float)
    radius = pd.to_numeric(pool["radius_p"], errors="coerce").to_numpy(float)
    puffy = radius > np.interp(mass, m_sil, r_sil)

    if population == "flat":
        td = run_kepler(pool)["detected"].to_numpy(bool)
    else:
        td = pd.to_numeric(pool["detected"], errors="coerce").fillna(0).astype(bool).to_numpy()
    rd = run_rv_best(pool, mag_target=RV_MAG_TARGET)["detected"].to_numpy(bool)
    det = td & rd
    return mass, radius, puffy, det


def mc_universe(mass, radius, puffy, det, drop_highmass_rocky, m_sil, r_sil, rng):
    keep = ~((~puffy) & (mass > MASS_THRESHOLD)) if drop_highmass_rocky else np.ones(len(mass), bool)
    idx = np.flatnonzero(keep)
    m_u, r_u, det_u = mass[idx], radius[idx], det[idx]
    L = idx.size

    pf = np.full(N_REPEATS, np.nan)
    dn = np.full(N_REPEATS, np.nan)
    cnt = np.zeros(N_REPEATS)
    for i in range(N_REPEATS):
        s = rng.integers(0, L, N_SAMPLE_PER_UNIVERSE)
        sel = s[det_u[s]]
        if sel.size == 0:
            continue
        mm, rr = m_u[sel], r_u[sel]
        m_obs = mm * np.exp(rng.normal(0.0, MASS_FRAC_ERR, sel.size))
        r_obs = rr * np.exp(rng.normal(0.0, RAD_FRAC_ERR, sel.size))
        pf[i], dn[i] = metrics(m_obs, r_obs, m_sil, r_sil)
        cnt[i] = sel.size

    # detected, un-noised reference value (the "true observed" each metric would give)
    if det_u.any():
        tp, tn = metrics(m_u[det_u], r_u[det_u], m_sil, r_sil)
    else:
        tp = tn = np.nan
    return {"puffy": pf, "dens": dn}, {"puffy": tp, "dens": tn}, float(np.nanmean(cnt))


def load_nasa_full():
    df = pd.read_csv(NASA_FILE)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    me1 = pd.to_numeric(df["pl_bmasseerr1"], errors="coerce").abs()
    me2 = pd.to_numeric(df["pl_bmasseerr2"], errors="coerce").abs()
    re1 = pd.to_numeric(df["pl_radeerr1"], errors="coerce").abs()
    re2 = pd.to_numeric(df["pl_radeerr2"], errors="coerce").abs()
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    measured = prov.str.contains("Mass|Msini", case=False, na=False) & ~prov.str.contains("Calc", case=False, na=False)
    keep = measured & r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    m, r = m[keep].to_numpy(), r[keep].to_numpy()
    me1, me2 = me1[keep].to_numpy(), me2[keep].to_numpy()
    re1, re2 = re1[keep].to_numpy(), re2[keep].to_numpy()
    # impute missing errors with the median fractional error
    me1 = np.where(np.isfinite(me1), me1, MASS_FRAC_ERR * m)
    me2 = np.where(np.isfinite(me2), me2, MASS_FRAC_ERR * m)
    re1 = np.where(np.isfinite(re1), re1, RAD_FRAC_ERR * r)
    re2 = np.where(np.isfinite(re2), re2, RAD_FRAC_ERR * r)
    return m, r, me1, me2, re1, re2


def mc_nasa(m, r, me1, me2, re1, re2, m_sil, r_sil, rng, with_errors):
    n = len(m)
    pf = np.empty(N_REPEATS); dn = np.empty(N_REPEATS)
    for i in range(N_REPEATS):
        b = rng.integers(0, n, n)
        mb, rb = m[b], r[b]
        if with_errors:
            zm, zr = rng.normal(size=n), rng.normal(size=n)
            mb = mb + np.where(zm >= 0, zm * me1[b], zm * me2[b])
            rb = rb + np.where(zr >= 0, zr * re1[b], zr * re2[b])
            mb = np.clip(mb, 1e-3, None); rb = np.clip(rb, 1e-3, None)
        pf[i], dn[i] = metrics(mb, rb, m_sil, r_sil)
    return {"puffy": pf, "dens": dn}, n


def tension(model_arr, nasa_arr):
    a = model_arr[np.isfinite(model_arr)]; b = nasa_arr[np.isfinite(nasa_arr)]
    dmu = abs(a.mean() - b.mean())
    sig = np.sqrt(a.std() ** 2 + b.std() ** 2)
    return dmu / sig if sig > 0 else np.inf


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) if sd > 0 else np.zeros_like(x)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    rng = np.random.default_rng(RNG_SEED)

    print(f"--> N_SAMPLE={N_SAMPLE_PER_UNIVERSE:,}  N_REPEATS={N_REPEATS:,}  "
          f"meas-noise mass={MASS_FRAC_ERR:.0%} radius={RAD_FRAC_ERR:.1%}  (transit+RV detection)")
    print("--> building pools + detectors...")
    pools = {pop: build_pool(pop, m_sil, r_sil) for pop in ["flat", "ppop"]}

    uni = []  # (label, colour, dict(metric->samples), dict(metric->true), neff)
    for pop, drop, label, colour in UNIVERSES:
        mass, radius, puffy, det = pools[pop]
        samp, true, neff = mc_universe(mass, radius, puffy, det, drop, m_sil, r_sil, rng)
        uni.append((label, colour, samp, true, neff))

    m, r, me1, me2, re1, re2 = load_nasa_full()
    nasa_err, n_nasa = mc_nasa(m, r, me1, me2, re1, re2, m_sil, r_sil, rng, with_errors=True)
    nasa_boot, _ = mc_nasa(m, r, me1, me2, re1, re2, m_sil, r_sil, rng, with_errors=False)

    # ---- closeness table ----
    print(f"\n  Tension to NASA (sigma; smaller = closer to observed).  NASA N={n_nasa}, "
          f"errors widen it from bootstrap-only.")
    header = "  " + f"{'universe':<26}" + "".join(f"{mk:>22}" for mk, _, _ in METRICS)
    print(header); print("  " + "-" * (len(header) - 2))
    winners = {}
    best = {mk: (None, np.inf) for mk, _, _ in METRICS}
    for label, colour, samp, true, neff in uni:
        cells = []
        for mk, _, fmt in METRICS:
            t = tension(samp[mk], nasa_err[mk])
            cells.append(f"mu={fmt.format(np.nanmean(samp[mk]))} ({t:.1f}s)")
            if t < best[mk][1]:
                best[mk] = (label, t)
        print(f"  {label:<26}" + "".join(f"{c:>22}" for c in cells))
    for mk, _, fmt in METRICS:
        winners[mk] = best[mk]
        print(f"    -> closest on {mk:<6}: {best[mk][0]}  ({best[mk][1]:.2f} sigma)")
    print(f"\n  NASA mu:  puffy={np.nanmean(nasa_err['puffy']):.2f}  "
          f"dens={np.nanmean(nasa_err['dens']):.2f}  "
          f"(N_det/draw, models: " + ", ".join(f"{lbl.split(':')[0]}={n:.0f}" for lbl, _, _, _, n in uni) + ")")

    # ---- plot: one panel per metric ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.6))
    for ax, (mk, xlabel, fmt) in zip(axes, METRICS):
        allv = [nasa_err[mk][np.isfinite(nasa_err[mk])], nasa_boot[mk][np.isfinite(nasa_boot[mk])]]
        allv += [s[mk][np.isfinite(s[mk])] for _, _, s, _, _ in uni]
        cat = np.concatenate(allv)
        lo, hi = cat.min(), cat.max()
        pad = 0.04 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
        edges = np.linspace(lo, hi, 70)
        xs = np.linspace(lo, hi, 500)

        for label, colour, samp, true, neff in uni:
            s = samp[mk][np.isfinite(samp[mk])]
            ax.hist(s, bins=edges, density=True, color=colour, alpha=0.26)
            tag = " <-- closest" if winners[mk][0] == label else ""
            ax.plot(xs, gauss(xs, s.mean(), s.std()), color=colour, lw=2.0,
                    label=f"{label}: μ={fmt.format(s.mean())} | N_det≈{neff:.0f}{tag}")
            if np.isfinite(true[mk]):
                ax.axvline(true[mk], color=colour, ls=":", lw=1.0, alpha=0.5)

        ne = nasa_err[mk][np.isfinite(nasa_err[mk])]
        nb = nasa_boot[mk][np.isfinite(nasa_boot[mk])]
        ax.hist(ne, bins=edges, density=True, color="tab:green", alpha=0.30)
        ax.plot(xs, gauss(xs, ne.mean(), ne.std()), color="tab:green", lw=2.4,
                label=f"NASA + meas.errors: μ={fmt.format(ne.mean())} | N={n_nasa}")
        ax.plot(xs, gauss(xs, nb.mean(), nb.std()), color="darkgreen", lw=1.4, ls="--",
                label=f"NASA bootstrap only: μ={fmt.format(nb.mean())}")
        ax.set_title(xlabel, fontsize=10)
        ax.set_xlabel(xlabel.split("  ")[0])
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7.6, loc="best")
    axes[0].set_ylabel("How often each value came up\n(probability density over the repeated draws)")

    fig.suptitle(
        "Which 'fake' universe is closest to the OBSERVED (NASA) one? — two metrics (curve-based + curve-free), error-propagated, transit+RV detected\n"
        f"models forward-modelled to the observed plane (detect → add NASA-like noise: mass {MASS_FRAC_ERR:.0%}, radius {RAD_FRAC_ERR:.1%}); "
        "dotted line = model's noise-free detected value; dashed green = NASA without measurement error",
        fontsize=11)
    fig.text(0.5, 0.005,
             "Propagating NASA's measurement errors WIDENS the green curve (solid vs dashed) and is the honest uncertainty. "
             "'Closest' = smallest tension |μ_model−μ_NASA|/√(σ²+σ²).  Bulk density (right) needs no silicate curve, so it is the "
             "curve-independent cross-check on the puffy fraction (left).",
             ha="center", fontsize=8.8)
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    out_png = os.path.join(OUT_DIR, "universe_metrics_vs_nasa.png")
    fig.savefig(out_png, dpi=185, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


if __name__ == "__main__":
    main()

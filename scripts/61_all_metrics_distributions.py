"""
61_all_metrics_distributions.py — the bell-curve (sampling-distribution) view of ALL the
script-58 metrics, in the style of script 57. One panel per metric; in each panel the four
universes (flat A/B, P-Pop A/B) and NASA are drawn as distributions over repeated draws.

Each realization: sample N_SAMPLE planets from the universe, keep the transit+RV detected
ones, add NASA-like measurement noise, compute every metric. Repeating N_REPEATS times gives
the bell. NASA = bootstrap + reported asymmetric measurement errors (same as script 57).
Dotted vertical line = the model's noise-free detected value; legend shows mu and tension(sigma).

Metrics:
  curve-dependent : puffy fraction, mean distance to silicate curve
  curve-free      : median bulk density, M-R slope, M-R intrinsic scatter [dex],
                    log-density scatter [dex], density bimodality (Sarle BC), density GMM dBIC

Run:
    python scripts/61_all_metrics_distributions.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # must precede numpy/sklearn: kills MKL KMeans leak warning

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
from run.ppop import universe_metrics as UM

SILICATE_CURVE = ROOT / "Hongyi-silicon.ddat"
PPOP_CATALOG = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = os.path.join(ROOT, "my_outputs", "61_all_metrics_distributions")

# ============================ KNOBS ============================
N_SAMPLE_PER_UNIVERSE = 20000   # planets grabbed per draw
N_REPEATS = 3000                # draws per universe / NASA bootstraps (raise for smoother bells)
MASS_THRESHOLD = 2.0
MASS_FRAC_ERR, RAD_FRAC_ERR = 0.20, 0.047
# ==============================================================

FLAT_N_POOL = 300000
RNG_SEED = 0
RV_MAG_TARGET = 12.0
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)

UNIVERSES = [
    ("flat", True,  "flat A: no rocky M>2", "tab:orange"),
    ("flat", False, "flat B: with rocky",   "tab:blue"),
    ("ppop", True,  "P-Pop A: no rocky M>2", "tab:red"),
    ("ppop", False, "P-Pop B: with rocky",   "tab:purple"),
]
# key, panel title, value format
METRICS = [
    ("puffy_frac",          "puffy fraction  (curve-dep)",              "{:.2f}"),
    ("dist_curve",          "mean dist to silicate curve [R⊕] (curve-dep)", "{:+.3f}"),
    ("med_dens",            "median bulk density [g/cc] (curve-free)",  "{:.2f}"),
    ("MR_slope",            "M-R slope  d logR / d logM (curve-free)",  "{:.2f}"),
    ("MR_scatter_dex",      "M-R intrinsic scatter [dex] (curve-free)", "{:.3f}"),
    ("logdens_scatter_dex", "log-density scatter [dex] (curve-free)",   "{:.3f}"),
    ("bimodality_BC",       "density bimodality, Sarle BC (curve-free)", "{:.2f}"),
    ("GMM_dBIC",            "density GMM ΔBIC, 2 vs 1 (curve-free)",     "{:+.0f}"),
]
KEYS = [m[0] for m in METRICS]


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def metric_vec(M, R, m_sil, r_sil):
    sl, _, sc = UM.mr_scatter(M, R)
    bc, dbic = UM.density_bimodality(M, R)
    return {
        "puffy_frac": UM.puffy_fraction(M, R, m_sil, r_sil),
        "dist_curve": UM.mean_distance_to_curve(M, R, m_sil, r_sil),
        "med_dens": UM.median_density(M, R),
        "MR_slope": sl,
        "MR_scatter_dex": sc,
        "logdens_scatter_dex": UM.logdensity_scatter(M, R),
        "bimodality_BC": bc,
        "GMM_dBIC": dbic,
    }


def build_pool(population, m_sil, r_sil):
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
    return mass, radius, puffy, td & rd


def mc_universe(mass, radius, puffy, det, drop, m_sil, r_sil, rng):
    keep = ~((~puffy) & (mass > MASS_THRESHOLD)) if drop else np.ones(len(mass), bool)
    idx = np.flatnonzero(keep)
    m_u, r_u, det_u = mass[idx], radius[idx], det[idx]
    L = idx.size
    out = {k: np.full(N_REPEATS, np.nan) for k in KEYS}
    neff = []
    for i in range(N_REPEATS):
        s = rng.integers(0, L, N_SAMPLE_PER_UNIVERSE)
        sel = s[det_u[s]]
        if sel.size < 20:
            continue
        Mo = m_u[sel] * np.exp(rng.normal(0, MASS_FRAC_ERR, sel.size))
        Ro = r_u[sel] * np.exp(rng.normal(0, RAD_FRAC_ERR, sel.size))
        mv = metric_vec(Mo, Ro, m_sil, r_sil)
        for k in KEYS:
            out[k][i] = mv[k]
        neff.append(sel.size)
    true = metric_vec(m_u[det_u], r_u[det_u], m_sil, r_sil) if det_u.sum() >= 20 else {k: np.nan for k in KEYS}
    return out, true, float(np.mean(neff)) if neff else 0.0


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
    me1 = np.where(np.isfinite(me1), me1, MASS_FRAC_ERR * m)
    me2 = np.where(np.isfinite(me2), me2, MASS_FRAC_ERR * m)
    re1 = np.where(np.isfinite(re1), re1, RAD_FRAC_ERR * r)
    re2 = np.where(np.isfinite(re2), re2, RAD_FRAC_ERR * r)
    return m, r, me1, me2, re1, re2


def mc_nasa(m, r, me1, me2, re1, re2, m_sil, r_sil, rng):
    n = len(m)
    out = {k: np.full(N_REPEATS, np.nan) for k in KEYS}
    for i in range(N_REPEATS):
        b = rng.integers(0, n, n)
        zm, zr = rng.normal(size=n), rng.normal(size=n)
        mb = np.clip(m[b] + np.where(zm >= 0, zm * me1[b], zm * me2[b]), 1e-3, None)
        rb = np.clip(r[b] + np.where(zr >= 0, zr * re1[b], zr * re2[b]), 1e-3, None)
        mv = metric_vec(mb, rb, m_sil, r_sil)
        for k in KEYS:
            out[k][i] = mv[k]
    return out, n


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) if sd > 0 else np.zeros_like(x)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    rng = np.random.default_rng(RNG_SEED)

    print(f"--> N_SAMPLE={N_SAMPLE_PER_UNIVERSE:,}  N_REPEATS={N_REPEATS:,}  (transit+RV detected, noise-propagated)")
    print("--> building pools + detectors...")
    pools = {pop: build_pool(pop, m_sil, r_sil) for pop in ["flat", "ppop"]}

    uni = []  # (label, colour, out_dict, true_dict, neff)
    for pop, drop, label, colour in UNIVERSES:
        mass, radius, puffy, det = pools[pop]
        out, true, neff = mc_universe(mass, radius, puffy, det, drop, m_sil, r_sil, rng)
        uni.append((label, colour, out, true, neff))

    nm, nr, me1, me2, re1, re2 = load_nasa_full()
    nasa, n_nasa = mc_nasa(nm, nr, me1, me2, re1, re2, m_sil, r_sil, rng)

    # ---- 2x4 panels ----
    fig, axes = plt.subplots(2, 4, figsize=(22, 10.5))
    axes = axes.ravel()
    for ax, (key, title, fmt) in zip(axes, METRICS):
        nval = nasa[key][np.isfinite(nasa[key])]
        n_mu, n_sd = nval.mean(), nval.std()
        allv = [nval] + [u[2][key][np.isfinite(u[2][key])] for u in uni]
        cat = np.concatenate([a for a in allv if a.size])
        lo, hi = cat.min(), cat.max()
        pad = 0.05 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
        edges = np.linspace(lo, hi, 60)
        xs = np.linspace(lo, hi, 500)

        for label, colour, out, true, neff in uni:
            s = out[key][np.isfinite(out[key])]
            if s.size == 0:
                continue
            tens = abs(s.mean() - n_mu) / n_sd if n_sd > 0 else np.nan
            ax.hist(s, bins=edges, density=True, color=colour, alpha=0.26)
            ax.plot(xs, gauss(xs, s.mean(), s.std()), color=colour, lw=1.9,
                    label=f"{label.split(':')[0]}: μ={fmt.format(s.mean())} ({tens:.1f}σ)")
            if np.isfinite(true[key]):
                ax.axvline(true[key], color=colour, ls=":", lw=1.0, alpha=0.5)
        ax.hist(nval, bins=edges, density=True, color="tab:green", alpha=0.32)
        ax.plot(xs, gauss(xs, n_mu, n_sd), color="tab:green", lw=2.3,
                label=f"NASA: μ={fmt.format(n_mu)} (N={n_nasa})")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7.0, loc="best")
    for ax in (axes[0], axes[4]):
        ax.set_ylabel("density over repeated draws")

    fig.suptitle(
        "All-metrics sampling distributions — four 'fake' universes vs OBSERVED (NASA), transit+RV detected, "
        "measurement-error propagated\n"
        f"each bell = {N_REPEATS:,} draws of {N_SAMPLE_PER_UNIVERSE:,} planets;  legend shows μ and tension to NASA (σ);  "
        "dotted line = model's noise-free detected value",
        fontsize=12)
    fig.text(0.5, 0.005,
             "Top row + median density = WHERE planets sit (P-Pop B closest). M-R/density SCATTER + bimodality = HOW diverse "
             "they are: NASA is wide, P-Pop B too tight, flat B matches the spread but at impossible density. "
             "Data want P-Pop B's location PLUS more intrinsic scatter.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    out_png = os.path.join(OUT_DIR, "all_metrics_distributions.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out_png}")


if __name__ == "__main__":
    main()

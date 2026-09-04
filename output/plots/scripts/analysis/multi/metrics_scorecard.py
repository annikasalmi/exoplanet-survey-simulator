"""
23_metrics_scorecard.py — run ALL the population-comparison metrics (not just the
puffy fraction) on the four universes vs NASA, error-propagated, transit+RV detected.

Metrics (from run/ppop/universe_metrics.py):
  curve-dependent : puffy fraction, mean distance to silicate curve
  curve-free      : median bulk density, M-R slope, M-R intrinsic scatter [dex],
                    log-density scatter [dex], density bimodality (Sarle BC, GMM dBIC)
  distribution    : 2-D energy distance to NASA in logM-logR (+ permutation p)
  likelihood      : KDE mean log-likelihood of NASA under each universe (-> Bayes factor)

Each universe is forward-modelled to the observed plane (transit+RV detect -> add NASA-like
measurement noise). NASA is bootstrapped WITH its reported asymmetric errors.

Output: a printed scorecard table + a tension bar chart.

Run:
    python scripts/23_metrics_scorecard.py
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # must precede numpy/sklearn: kills MKL KMeans leak warning

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
OUT_DIR = os.path.join(ROOT, "output/plots", "23_metrics_scorecard")

MASS_THRESHOLD = 2.0
MASS_FRAC_ERR = 0.20
RAD_FRAC_ERR = 0.047
N_NASA_BOOT = 400          # NASA bootstraps for scalar-metric uncertainty
MODEL_OBS_CAP = 4000       # cap on model detected sample used for metrics
RNG_SEED = 0
FLAT_N_POOL = 300000
RV_MAG_TARGET = 12.0
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)

UNIVERSES = [
    ("flat", True,  "flat A", "tab:orange"),
    ("flat", False, "flat B", "tab:blue"),
    ("ppop", True,  "P-Pop A", "tab:red"),
    ("ppop", False, "P-Pop B", "tab:purple"),
]


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


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
    if population == "flat":
        td = run_kepler(pool)["detected"].to_numpy(bool)
    else:
        td = pd.to_numeric(pool["detected"], errors="coerce").fillna(0).astype(bool).to_numpy()
    rd = run_rv_best(pool, mag_target=RV_MAG_TARGET)["detected"].to_numpy(bool)
    return mass, radius, td & rd


def observed_sample(mass, radius, det, drop_highmass_rocky, m_sil, r_sil, rng):
    keep = np.ones(len(mass), bool)
    if drop_highmass_rocky:
        puffy = radius > np.interp(mass, m_sil, r_sil)
        keep = ~((~puffy) & (mass > MASS_THRESHOLD))
    idx = np.flatnonzero(keep & det)
    if idx.size > MODEL_OBS_CAP:
        idx = rng.choice(idx, MODEL_OBS_CAP, replace=False)
    m, r = mass[idx], radius[idx]
    m_obs = m * np.exp(rng.normal(0, MASS_FRAC_ERR, m.size))
    r_obs = r * np.exp(rng.normal(0, RAD_FRAC_ERR, r.size))
    return m_obs, r_obs


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


def nasa_realization(m, r, me1, me2, re1, re2, rng):
    n = len(m)
    b = rng.integers(0, n, n)
    zm, zr = rng.normal(size=n), rng.normal(size=n)
    mb = np.clip(m[b] + np.where(zm >= 0, zm * me1[b], zm * me2[b]), 1e-3, None)
    rb = np.clip(r[b] + np.where(zr >= 0, zr * re1[b], zr * re2[b]), 1e-3, None)
    return mb, rb


# scalar metrics computed on one (M,R) sample
def scalars(M, R, m_sil, r_sil):
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


SCALAR_KEYS = ["puffy_frac", "dist_curve", "med_dens", "MR_slope", "MR_scatter_dex",
               "logdens_scatter_dex", "bimodality_BC", "GMM_dBIC"]
SCALAR_FMT = {"puffy_frac": "{:.2f}", "dist_curve": "{:+.3f}", "med_dens": "{:.2f}",
              "MR_slope": "{:.2f}", "MR_scatter_dex": "{:.3f}", "logdens_scatter_dex": "{:.3f}",
              "bimodality_BC": "{:.2f}", "GMM_dBIC": "{:+.0f}"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    rng = np.random.default_rng(RNG_SEED)

    print("--> building pools + detectors (transit+RV)...")
    pools = {pop: build_pool(pop, m_sil, r_sil) for pop in ["flat", "ppop"]}
    nm, nr, me1, me2, re1, re2 = load_nasa_full()
    n_nasa = len(nm)

    # NASA scalar metrics: bootstrap with errors
    nasa_runs = {k: [] for k in SCALAR_KEYS}
    nasa_obs_M, nasa_obs_R = [], []
    for _ in range(N_NASA_BOOT):
        mb, rb = nasa_realization(nm, nr, me1, me2, re1, re2, rng)
        s = scalars(mb, rb, m_sil, r_sil)
        for k in SCALAR_KEYS:
            nasa_runs[k].append(s[k])
        if len(nasa_obs_M) < 3000:
            nasa_obs_M.append(mb); nasa_obs_R.append(rb)
    nasa_mu = {k: float(np.nanmean(nasa_runs[k])) for k in SCALAR_KEYS}
    nasa_sd = {k: float(np.nanstd(nasa_runs[k])) for k in SCALAR_KEYS}
    nasa_obs_M = np.concatenate(nasa_obs_M); nasa_obs_R = np.concatenate(nasa_obs_R)

    # model universes
    rows = {}      # label -> dict(scalar values)
    tensions = {}  # label -> dict(metric -> sigma)
    dist_metrics = {}  # label -> (energy_dist, energy_p, kde_loglik)
    for pop, drop, label, colour in UNIVERSES:
        mass, radius, det = pools[pop]
        M, R = observed_sample(mass, radius, det, drop, m_sil, r_sil, rng)
        s = scalars(M, R, m_sil, r_sil)
        rows[label] = s
        tensions[label] = {k: (abs(s[k] - nasa_mu[k]) / nasa_sd[k] if nasa_sd[k] > 0 else np.nan)
                           for k in SCALAR_KEYS}
        ed, ep = UM.energy_distance(M, R, nm, nr, n_perm=200, rng=rng)
        ll = UM.kde_mean_loglik(M, R, nasa_obs_M, nasa_obs_R)
        dist_metrics[label] = (ed, ep, ll)

    # ---- print scorecard ----
    labels = [l for _, _, l, _ in UNIVERSES]
    print(f"\n  ============ SCORECARD (NASA N={n_nasa}; transit+RV detected; meas-noise propagated) ============")
    print(f"  {'metric':<22}" + "".join(f"{l:>12}" for l in labels) + f"{'NASA':>12}")
    print("  " + "-" * (22 + 12 * (len(labels) + 1)))
    for k in SCALAR_KEYS:
        f = SCALAR_FMT[k]
        line = f"  {k:<22}" + "".join(f"{f.format(rows[l][k]):>12}" for l in labels)
        line += f"{f.format(nasa_mu[k]):>12}"
        print(line)
        print(f"  {'  tension(sigma)':<22}" + "".join(f"{tensions[l][k]:>12.1f}" for l in labels) + f"{'--':>12}")
    print("  " + "-" * (22 + 12 * (len(labels) + 1)))
    print(f"  {'energy dist -> NASA':<22}" + "".join(f"{dist_metrics[l][0]:>12.3f}" for l in labels) + f"{'0 (self)':>12}")
    print(f"  {'  (perm p-value)':<22}" + "".join(f"{dist_metrics[l][1]:>12.3f}" for l in labels))
    print(f"  {'KDE mean loglik NASA':<22}" + "".join(f"{dist_metrics[l][2]:>12.3f}" for l in labels))

    # Bayes factor vs best
    lls = {l: dist_metrics[l][2] for l in labels}
    best = max(lls, key=lambda l: lls[l])
    print(f"\n  Most NASA-like by every distance/likelihood metric -> closest energy dist & highest KDE loglik:")
    print(f"    energy-dist winner : {min(labels, key=lambda l: dist_metrics[l][0])}")
    print(f"    KDE-loglik winner  : {best}")
    print(f"    log Bayes factor (best vs others), = N_NASA*(ll_best - ll_other):")
    for l in labels:
        if l != best:
            print(f"      {best} vs {l:<8}: lnB = {n_nasa * (lls[best] - lls[l]):+.0f}")

    # per-metric closest universe (scalars by tension)
    print("\n  Closest universe per scalar metric (min tension):")
    for k in SCALAR_KEYS:
        w = min(labels, key=lambda l: tensions[l][k] if np.isfinite(tensions[l][k]) else np.inf)
        print(f"    {k:<22}: {w}  ({tensions[w][k]:.2f} sigma)")

    # ---- figure: tension bar chart ----
    fig, ax = plt.subplots(figsize=(15, 7))
    keys = SCALAR_KEYS
    x = np.arange(len(keys))
    w = 0.2
    colours = {l: c for _, _, l, c in UNIVERSES}
    for i, l in enumerate(labels):
        vals = [min(tensions[l][k], 12) if np.isfinite(tensions[l][k]) else 0 for k in keys]
        ax.bar(x + (i - 1.5) * w, vals, w, label=l, color=colours[l], alpha=0.85)
    ax.axhline(2, color="k", ls="--", lw=1, alpha=0.6)
    ax.text(len(keys) - 0.5, 2.15, "2σ (consistent below)", fontsize=8, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("tension to NASA  [σ]   (lower = closer to observed; capped at 12)")
    ax.set_title("All-metrics scorecard — how far each 'fake' universe sits from OBSERVED (NASA)\n"
                 "transit+RV detected, measurement-error propagated; bars below 2σ = statistically consistent",
                 fontsize=12)
    ax.legend(title="universe", fontsize=10)
    ax.grid(alpha=0.2, axis="y")
    # annotate distribution metrics
    txt = "distribution / likelihood metrics:\n" + "\n".join(
        f"  {l}: energyD={dist_metrics[l][0]:.2f} (p={dist_metrics[l][1]:.2f}), KDE_ll={dist_metrics[l][2]:.2f}"
        for l in labels)
    ax.text(0.012, 0.97, txt, transform=ax.transAxes, va="top", fontsize=8.5,
            family="monospace", bbox=dict(boxstyle="round", fc="whitesmoke", ec="0.7"))
    fig.tight_layout()
    out_png = os.path.join(OUT_DIR, "all_metrics_scorecard.png")
    fig.savefig(out_png, dpi=185, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


if __name__ == "__main__":
    main()

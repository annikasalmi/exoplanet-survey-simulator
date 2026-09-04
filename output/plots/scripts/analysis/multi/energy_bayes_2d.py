"""
33_energy_bayes_2d.py — the two PROPER whole-distribution model comparisons in the 2-D
mass-radius (logM-logR) plane, for the four universes vs NASA. No 1-D projection, no
silicate curve.

(A) 2-D ENERGY DISTANCE, shown as a distribution.
    "mean distance to silicate curve" is NOT this — that is a 1-D signed gap to a curve.
    The energy distance is a true 2-sample distance between two point clouds in (logM,logR).
    Per draw: take a NASA-sized observed sample of a universe and a NASA error-propagated
    sample, compute the energy distance. The NASA-vs-NASA "null" is what you'd get if the
    universe WERE NASA (finite-sample floor). A universe whose bell overlaps the null is
    statistically indistinguishable from NASA in 2-D.

(B) ERROR-CONVOLVED BAYES FACTOR.
    Build a Gaussian KDE of each universe's detected-observed (M,R). For each NASA planet,
    the likelihood is the model density CONVOLVED with that planet's measurement-error kernel
    (Monte-Carlo over the error ellipse in logM-logR). Sum log over planets -> total log-
    likelihood; differences are log Bayes factors -> "universe X is 10^k times more likely".
    (Caveat: likelihood rewards a model that simply COVERS the data, so a too-diffuse model
    can win by hedging — read it together with the energy distance, which rewards location.)

Run:
    python scripts/33_energy_bayes_2d.py
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
from scipy.stats import gaussian_kde

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
OUT_DIR = os.path.join(ROOT, "output/plots", "33_energy_bayes_2d")

MASS_THRESHOLD = 2.0
MASS_FRAC_ERR, RAD_FRAC_ERR = 0.20, 0.047
K_ED = 600            # realizations for the energy-distance distribution
N_DRAW = 292          # NASA-sized samples
KDE_TRAIN_CAP = 2500  # model points used to build each KDE
S_ERR = 200           # error-kernel Monte-Carlo samples per NASA planet
FLAT_N_POOL = 300000
RNG_SEED = 0
RV_MAG_TARGET = 12.0
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)
LN10 = np.log(10.0)

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
    puffy = radius > np.interp(mass, m_sil, r_sil)
    if population == "flat":
        td = run_kepler(pool)["detected"].to_numpy(bool)
    else:
        td = pd.to_numeric(pool["detected"], errors="coerce").fillna(0).astype(bool).to_numpy()
    rd = run_rv_best(pool, mag_target=RV_MAG_TARGET)["detected"].to_numpy(bool)
    det = td & rd
    keepU = lambda drop: (~((~puffy) & (mass > MASS_THRESHOLD)) if drop else np.ones(len(mass), bool))
    return mass, radius, det, keepU


def load_nasa():
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
    me1 = np.where(np.isfinite(me1), me1, MASS_FRAC_ERR * m); me2 = np.where(np.isfinite(me2), me2, MASS_FRAC_ERR * m)
    re1 = np.where(np.isfinite(re1), re1, RAD_FRAC_ERR * r); re2 = np.where(np.isfinite(re2), re2, RAD_FRAC_ERR * r)
    return m, r, me1, me2, re1, re2


def nasa_obs(m, r, me1, me2, re1, re2, rng, n=None):
    N = len(m); n = n or N
    b = rng.integers(0, N, n)
    zm, zr = rng.normal(size=n), rng.normal(size=n)
    mb = np.clip(m[b] + np.where(zm >= 0, zm * me1[b], zm * me2[b]), 1e-3, None)
    rb = np.clip(r[b] + np.where(zr >= 0, zr * re1[b], zr * re2[b]), 1e-3, None)
    return mb, rb


def uni_obs(mass, radius, det, keep, rng, n):
    idx = np.flatnonzero(keep & det)
    j = idx[rng.integers(0, idx.size, n)]
    Mo = mass[j] * np.exp(rng.normal(0, MASS_FRAC_ERR, n))
    Ro = radius[j] * np.exp(rng.normal(0, RAD_FRAC_ERR, n))
    return Mo, Ro


def _mpd(X, Y):
    return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)).mean()


def energy_distance_fixed(M1, R1, M2, R2, mu, sd):
    A = (np.column_stack([np.log10(M1), np.log10(R1)]) - mu) / sd
    B = (np.column_stack([np.log10(M2), np.log10(R2)]) - mu) / sd
    return 2 * _mpd(A, B) - _mpd(A, A) - _mpd(B, B)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    rng = np.random.default_rng(RNG_SEED)
    print("--> building pools + detectors...")
    pools = {pop: build_pool(pop, m_sil, r_sil) for pop in ["flat", "ppop"]}
    nm, nr, me1, me2, re1, re2 = load_nasa()
    n_nasa = len(nm)

    # fixed standardisation scale from NASA
    mu = np.array([np.log10(nm).mean(), np.log10(nr).mean()])
    sd = np.array([np.log10(nm).std(), np.log10(nr).std()])

    # ---------- (A) energy-distance distributions ----------
    print(f"--> (A) energy distance, {K_ED} realizations...")
    ed = {label: np.empty(K_ED) for _, _, label, _ in UNIVERSES}
    ed_null = np.empty(K_ED)
    for k in range(K_ED):
        nM, nR = nasa_obs(nm, nr, me1, me2, re1, re2, rng, N_DRAW)
        for pop, drop, label, _ in UNIVERSES:
            mass, radius, det, keepU = pools[pop]
            uM, uR = uni_obs(mass, radius, det, keepU(drop), rng, N_DRAW)
            ed[label][k] = energy_distance_fixed(uM, uR, nM, nR, mu, sd)
        nM2, nR2 = nasa_obs(nm, nr, me1, me2, re1, re2, rng, N_DRAW)
        ed_null[k] = energy_distance_fixed(nM, nR, nM2, nR2, mu, sd)

    print(f"  {'universe':<10}{'mean energy dist':>18}{'vs NASA-null':>16}")
    print(f"  {'NASA null':<10}{ed_null.mean():>18.3f}{'(floor)':>16}")
    for _, _, label, _ in UNIVERSES:
        sep = (ed[label].mean() - ed_null.mean()) / np.sqrt(ed[label].std() ** 2 + ed_null.std() ** 2)
        print(f"  {label:<10}{ed[label].mean():>18.3f}{sep:>14.1f}σ")

    # ---------- (B) error-convolved Bayes factor ----------
    print(f"\n--> (B) error-convolved KDE likelihood ({S_ERR} error-samples/planet)...")
    nlogM, nlogR = np.log10(nm), np.log10(nr)
    sig_logM = (0.5 * (me1 + me2) / nm) / LN10
    sig_logR = (0.5 * (re1 + re2) / nr) / LN10
    # smeared NASA points (n_nasa x S_ERR) in logM-logR
    aM = (nlogM[:, None] + rng.normal(size=(n_nasa, S_ERR)) * sig_logM[:, None]).ravel()
    aR = (nlogR[:, None] + rng.normal(size=(n_nasa, S_ERR)) * sig_logR[:, None]).ravel()
    eval_pts = np.vstack([aM, aR])

    total_ll = {}
    for pop, drop, label, _ in UNIVERSES:
        mass, radius, det, keepU = pools[pop]
        uM, uR = uni_obs(mass, radius, det, keepU(drop), rng, min(KDE_TRAIN_CAP,
                                                                   int((keepU(drop) & det).sum())))
        train = np.vstack([np.log10(uM), np.log10(uR)])
        kde = gaussian_kde(train)
        dens = kde.pdf(eval_pts).reshape(n_nasa, S_ERR).mean(axis=1)
        total_ll[label] = float(np.log(np.clip(dens, 1e-300, None)).sum())

    best = max(total_ll, key=lambda l: total_ll[l])
    print(f"  {'universe':<10}{'total logL(NASA)':>18}{'ln B vs best':>16}{'~times more likely':>22}")
    for _, _, label, _ in UNIVERSES:
        lnB = total_ll[best] - total_ll[label]
        times = "best" if label == best else f"10^{lnB/LN10:.0f} less"
        print(f"  {label:<10}{total_ll[label]:>18.1f}{(-lnB):>16.1f}{times:>22}")
    print(f"  -> highest-likelihood universe: {best}")
    edist_win = min(ed, key=lambda l: ed[l].mean())
    print(f"  -> smallest 2-D energy distance: {edist_win}")

    # ---------- figure ----------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(17, 6.6))
    allv = np.concatenate([ed_null] + [ed[l] for _, _, l, _ in UNIVERSES])
    lo, hi = allv.min(), allv.max(); pad = 0.05 * (hi - lo)
    edges = np.linspace(lo - pad, hi + pad, 55); xs = np.linspace(lo - pad, hi + pad, 400)

    def g(x, s):
        return np.exp(-0.5 * ((x - s.mean()) / s.std()) ** 2) / (s.std() * np.sqrt(2 * np.pi))
    for pop, drop, label, colour in UNIVERSES:
        s = ed[label]
        axA.hist(s, bins=edges, density=True, color=colour, alpha=0.28)
        axA.plot(xs, g(xs, s), color=colour, lw=2, label=f"{label}: {s.mean():.3f}")
    axA.hist(ed_null, bins=edges, density=True, color="tab:green", alpha=0.4)
    axA.plot(xs, g(xs, ed_null), color="tab:green", lw=2.6, label=f"NASA vs NASA (null): {ed_null.mean():.3f}")
    axA.set_xlabel("2-D energy distance to NASA in (logM, logR)  — smaller = closer")
    axA.set_ylabel("density over draws")
    axA.set_title("(A) 2-D energy-distance test\n(overlap with green null = indistinguishable from NASA)", fontsize=11)
    axA.legend(fontsize=9, title="universe : mean")
    axA.grid(alpha=0.2)

    labels = [l for _, _, l, _ in UNIVERSES]
    colours = [c for _, _, _, c in UNIVERSES]
    vals = [total_ll[l] for l in labels]
    axB.bar(labels, vals, color=colours, alpha=0.85)
    axB.set_ylabel("total log-likelihood of NASA  (higher = more likely)")
    axB.set_title("(B) error-convolved KDE likelihood\nbar gaps = log Bayes factors", fontsize=11)
    axB.grid(alpha=0.2, axis="y")
    vmin = min(vals)
    for i, l in enumerate(labels):
        lnB = total_ll[best] - total_ll[l]
        txt = "best" if l == best else f"10^{lnB/LN10:.0f}× less"
        axB.text(i, vals[i], f"{vals[i]:.0f}\n{txt}", ha="center",
                 va="bottom" if vals[i] >= 0 else "top", fontsize=8.5)
    axB.set_ylim(vmin * 1.12 if vmin < 0 else 0, max(vals) * 0.9 if max(vals) < 0 else max(vals) * 1.1)

    fig.suptitle("Proper 2-D model comparison in mass-radius — energy distance (location) and error-convolved Bayes factor (coverage)",
                 fontsize=12)
    fig.text(0.5, 0.005,
             "Energy distance rewards being in the right PLACE; the KDE likelihood rewards COVERING the data (a too-diffuse "
             "model can win by hedging). Read both: the universe that is BOTH near the null AND high-likelihood is the honest best.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    out_png = os.path.join(OUT_DIR, "energy_bayes_2d.png")
    fig.savefig(out_png, dpi=185, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


if __name__ == "__main__":
    main()

"""
18_puffy_detection_flat.py — how each telescope (transit / RV / transit+RV) shifts the
puffy fraction, with the FLAT and P-Pop universes on separate rows.

Same method as script 56 (INTRINSIC detected puffy fraction: sample the true population, keep the
detected ones, measure N_puffy/(N_puffy+N_rocky) on their TRUE labels -- NO measurement noise here;
this isolates the DETECTION effect). NASA = bootstrap of the observed planets, same in every panel.

Layout 2x4:
    columns = Before any detector (true population) | Transit only | RV only | Transit AND RV
    row 1   = FLAT  universe (flat A, flat B)
    row 2   = P-Pop universe (P-Pop A, P-Pop B)
Dotted vertical line = each universe's TRUE (undetected) puffy fraction; the bell is where the
detector pushes it.

Run:
    python scripts/18_puffy_detection_flat.py
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
OUT_DIR = os.path.join(ROOT, "my_outputs", "18_puffy_detection_flat")

N_SAMPLE = 20000
N_REPEATS = 10000
MASS_THRESHOLD = 2.0
FLAT_N_POOL = 300000
RNG_SEED = 0
RV_MAG_TARGET = 12.0
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)

MODES = [("Before any detector\n(true population)", "none"),
         ("Transit only\n(Kepler)", "transit"),
         ("RV only\n(best HARPS/NIRPS)", "rv"),
         ("Transit AND RV", "both")]
ROWS = [("flat", "FLAT universe", [(True, "flat A", "tab:orange"), (False, "flat B", "tab:blue")]),
        ("ppop", "P-Pop universe", [(True, "P-Pop A", "tab:red"), (False, "P-Pop B", "tab:purple")])]


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def puffy_flag(mass, radius, m_sil, r_sil):
    return np.asarray(radius, float) > np.interp(np.asarray(mass, float), m_sil, r_sil)


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
    return mass, radius, puffy, td, rd


def det_for(td, rd, mode):
    if mode == "none":
        return np.ones(len(td), bool)   # before any detector: keep every planet
    if mode == "transit":
        return td
    if mode == "rv":
        return rd
    return td & rd


def mc_draws(mass, puffy, det, drop, rng):
    """Intrinsic detected puffy fraction (no measurement noise), chunked like script 56."""
    keep = ~((~puffy) & (mass > MASS_THRESHOLD)) if drop else np.ones(len(puffy), bool)
    idx = np.flatnonzero(keep)
    pk, dk = puffy[idx], det[idx]
    L = idx.size
    out = np.full(N_REPEATS, np.nan)
    cnt = np.empty(N_REPEATS)
    CH, pos = 400, 0
    while pos < N_REPEATS:
        c = min(CH, N_REPEATS - pos)
        S = rng.integers(0, L, size=(c, N_SAMPLE))
        dks = dk[S]; den = dks.sum(1); num = np.logical_and(pk[S], dks).sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[pos:pos + c] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        cnt[pos:pos + c] = den
        pos += c
    true_p = float(pk.mean())                       # undetected (dotted line)
    det_p = float(pk[dk].mean()) if dk.any() else np.nan
    return out[np.isfinite(out)], true_p, det_p, float(np.nanmean(cnt))


def load_nasa(m_sil, r_sil):
    df = pd.read_csv(NASA_FILE)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    meas = prov.str.contains("Mass|Msini", case=False, na=False) & ~prov.str.contains("Calc", case=False, na=False)
    keep = meas & r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
    mm, rr = m[keep].to_numpy(float), r[keep].to_numpy(float)
    puffy = puffy_flag(mm, rr, m_sil, r_sil)
    n = puffy.size
    S = np.random.default_rng(123).integers(0, n, size=(N_REPEATS, n))
    return puffy[S].mean(1), float(puffy.mean()), n


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi)) if sd > 0 else np.zeros_like(x)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    rng = np.random.default_rng(RNG_SEED)
    print("--> building pools + detectors...")
    pools = {pop: build_pool(pop, m_sil, r_sil) for pop in ["flat", "ppop"]}
    na, na_pt, n_nasa = load_nasa(m_sil, r_sil)
    na_mu, na_sd = na.mean(), na.std()

    fig, axes = plt.subplots(2, 4, figsize=(23, 10), sharex=True)
    print(f"\n  {'row':<7}{'mode':<10}{'universe':<9}{'true_p':>8}{'det μ':>8}{'σ':>8}{'N_det':>8}")
    for ri, (pop, row_label, models) in enumerate(ROWS):
        mass, radius, puffy, td, rd = pools[pop]
        for ci, (mode_label, mode) in enumerate(MODES):
            ax = axes[ri, ci]
            det = det_for(td, rd, mode)
            series = []
            for drop, label, colour in models:
                s, true_p, det_p, neff = mc_draws(mass, puffy, det, drop, rng)
                series.append((label, colour, s, true_p, neff))
                print(f"  {pop:<7}{mode:<10}{label:<9}{true_p:>8.3f}{det_p:>8.3f}{s.std():>8.3f}{neff:>8.0f}")
            allv = [na] + [s for _, _, s, _, _ in series if s.size]
            cat = np.concatenate([a for a in allv if a.size])
            lo, hi = cat.min(), cat.max(); pad = 0.06 * (hi - lo + 1e-9)
            edges = np.linspace(lo - pad, hi + pad, 60); xs = np.linspace(lo - pad, hi + pad, 400)
            for label, colour, s, true_p, neff in series:
                if s.size == 0:
                    continue
                ax.hist(s, bins=edges, density=True, color=colour, alpha=0.30)
                ax.plot(xs, gauss(xs, s.mean(), s.std()), color=colour, lw=2.0,
                        label=f"{label}: μ={s.mean():.2f} | N_det≈{neff:.0f}")
                ax.axvline(true_p, color=colour, ls=":", lw=1.1, alpha=0.55)
            ax.hist(na, bins=edges, density=True, color="tab:green", alpha=0.34)
            ax.plot(xs, gauss(xs, na_mu, na_sd), color="tab:green", lw=2.4,
                    label=f"NASA: μ={na_mu:.2f} (N={n_nasa})")
            ax.axvline(na_pt, color="darkgreen", ls=":", lw=1.2)
            ax.set_title((f"{mode_label}\n" if ri == 0 else "") + f"{row_label}", fontsize=10)
            ax.grid(alpha=0.2); ax.legend(fontsize=8, loc="upper left")
            if ci == 0:
                ax.set_ylabel("density over repeated draws")
            if ri == 1:
                ax.set_xlabel("puffy fraction  N_puffy/(N_puffy+N_rocky)")
        print()

    fig.suptitle("Puffy fraction: before any detector vs after each telescope — FLAT (top) vs P-Pop (bottom)\n"
                 "leftmost column = true population (no detector, N=20,000 → sharp); then transit / RV / transit+RV. "
                 "Dotted = true undetected value; green = NASA observed (no measurement noise applied)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = os.path.join(OUT_DIR, "puffy_detection_flat_ppop.png")
    fig.savefig(out_png, dpi=175, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out_png}")


if __name__ == "__main__":
    main()

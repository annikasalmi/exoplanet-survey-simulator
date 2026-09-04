"""
35_corner_occupancy_poisson.py — TEST A: completeness-corrected occupancy of the cold
super-Earth corner (rocky, mass>2 M⊕, insolation<50 I⊕).

Question: does NASA hold fewer cold large ROCKY planets than detection alone predicts?
  1. D = joint transit+RV detected fraction, measured on the FLAT universe (M ⊥ R ⊥ orbit
     by construction, so no occurrence prior enters D).
  2. Null "composition does not care about insolation": rocky M>2 planets are as common
     (per flat-measure unit) in the cold as in the hot. The unknown absolute occurrence is
     calibrated on NASA's HOT rocky M>2 count, per star stratum (FGK / M by Teff):
         λ(corner) = Σ_s  N_NASA(hot rocky, s) · C_flat_det(cold, s) / C_flat_det(hot, s)
  3. Poisson: p = P(X ≤ N_obs | λ).  Internal control: same machinery on the PUFFY class —
     if puffy is consistent while rocky shows a deficit, the effect is composition-specific.

Decision rule (inverse-detection-efficiency logic, Burke et al. 2015; Dressing &
Charbonneau 2015): λ >> N_obs with small p → deficit beyond selection (formation/retention);
λ ≈ 0 (D→0 in the corner) → emptiness is selection, formation unconstrained.

Run:
    python scripts/35_corner_occupancy_poisson.py
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
from scipy.stats import binomtest, poisson

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from run.ppop.uniform_generator import generate_flat_catalog
from run.ppop.flat_detect import run_kepler, run_rv_best

SILICATE_CURVE = Path(SILICON_CURVE)
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = os.path.join(ROOT, "output/plots", "35_corner_occupancy_poisson")

FLAT_N_POOL = 300000
RNG_SEED = 0
RV_MAG_TARGET = 12.0
MASS_MIN = 2.0
COLD_MAX = 50.0
NASA_MASS_PREC = 0.25
NASA_RAD_PREC = 0.08
BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)

STRATA = [("FGK", 3700.0, 7500.0), ("M", 0.0, 3700.0)]   # by host Teff [K]
FLUX_EDGES = np.logspace(-2, 4, 13)
RADIUS_EDGES = np.linspace(0.5, 2.2, 11)
MIN_CELL = 20


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def build_flat():
    pool = generate_flat_catalog(n_planets=FLAT_N_POOL, seed=RNG_SEED)
    r = pd.to_numeric(pool["radius_p"], errors="coerce")
    m = pd.to_numeric(pool["mass_p"], errors="coerce")
    f = pd.to_numeric(pool["flux_p"], errors="coerce")
    keep = (r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
            & f.between(BOX["f_lo"], BOX["f_hi"]))
    pool = pool[keep].copy()
    td = run_kepler(pool)["detected"].to_numpy(bool)
    rd = run_rv_best(pool, mag_target=RV_MAG_TARGET)["detected"].to_numpy(bool)
    return dict(
        mass=pd.to_numeric(pool["mass_p"], errors="coerce").to_numpy(float),
        radius=pd.to_numeric(pool["radius_p"], errors="coerce").to_numpy(float),
        flux=pd.to_numeric(pool["flux_p"], errors="coerce").to_numpy(float),
        teff=pd.to_numeric(pool["teff_s"], errors="coerce").to_numpy(float),
        det=td & rd,
    )


def load_nasa(precision: bool):
    df = pd.read_csv(NASA_FILE)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    ins = pd.to_numeric(df["pl_insol"], errors="coerce")
    teff = pd.to_numeric(df["st_teff"], errors="coerce")
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    meas = prov.str.contains("Mass|Msini", case=False, na=False) & ~prov.str.contains("Calc", case=False, na=False)
    keep = (meas & r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
            & ins.between(BOX["f_lo"], BOX["f_hi"]))
    if precision:
        me = np.maximum(pd.to_numeric(df["pl_bmasseerr1"], errors="coerce").abs(),
                        pd.to_numeric(df["pl_bmasseerr2"], errors="coerce").abs())
        re = np.maximum(pd.to_numeric(df["pl_radeerr1"], errors="coerce").abs(),
                        pd.to_numeric(df["pl_radeerr2"], errors="coerce").abs())
        keep = keep & (me / m <= NASA_MASS_PREC) & (re / r <= NASA_RAD_PREC)
    n_noteff = int((keep & ~teff.between(STRATA[1][1], STRATA[0][2])).sum())
    keep = keep & teff.between(STRATA[1][1], STRATA[0][2])
    print(f"    NASA ({'precision-cut' if precision else 'full measured'}): "
          f"{int(keep.sum())} planets in box/strata ({n_noteff} dropped: Teff missing or outside FGK/M)")
    return dict(m=m[keep].to_numpy(), r=r[keep].to_numpy(),
                ins=ins[keep].to_numpy(), teff=teff[keep].to_numpy())


def occupancy(cls_label, cls_flat, cls_nasa, flat, nasa):
    """Per-stratum calibration on NASA hot counts; returns totals + per-stratum rows."""
    rows, lam, var, naive, nobs = [], 0.0, 0.0, 0.0, 0
    for name, tlo, thi in STRATA:
        fs = (flat["teff"] >= tlo) & (flat["teff"] < thi) & cls_flat & (flat["mass"] > MASS_MIN)
        cold_f, hot_f = fs & (flat["flux"] < COLD_MAX), fs & (flat["flux"] >= COLD_MAX)
        c_cold = int((cold_f & flat["det"]).sum()); c_hot = int((hot_f & flat["det"]).sum())
        t_cold = int(cold_f.sum()); t_hot = int(hot_f.sum())
        ns = (nasa["teff"] >= tlo) & (nasa["teff"] < thi) & cls_nasa & (nasa["m"] > MASS_MIN)
        n_cold = int((ns & (nasa["ins"] < COLD_MAX)).sum())
        n_hot = int((ns & (nasa["ins"] >= COLD_MAX)).sum())
        ratio = c_cold / c_hot if c_hot > 0 else np.nan
        lam_s = n_hot * ratio if np.isfinite(ratio) else 0.0
        sig_s = (lam_s * np.sqrt(1.0 / c_cold + 1.0 / c_hot + 1.0 / n_hot)
                 if min(c_cold, c_hot, n_hot) > 0 else 0.0)
        naive_s = n_hot * (t_cold / t_hot) if t_hot > 0 else 0.0
        rows.append(dict(stratum=name, D_cold=c_cold / t_cold if t_cold else np.nan,
                         D_hot=c_hot / t_hot if t_hot else np.nan, c_cold=c_cold, c_hot=c_hot,
                         ratio=ratio, n_hot=n_hot, lam=lam_s, sig=sig_s, naive=naive_s, n_obs=n_cold))
        lam += lam_s; var += sig_s ** 2; naive += naive_s; nobs += n_cold
    sig = np.sqrt(var)
    p = float(poisson.cdf(nobs, lam)) if lam > 0 else np.nan
    p_lo = float(poisson.cdf(nobs, max(lam - sig, 1e-9))) if lam > 0 else np.nan
    p_hi = float(poisson.cdf(nobs, lam + sig)) if lam > 0 else np.nan
    return dict(label=cls_label, rows=rows, lam=lam, sig=sig, naive=naive, nobs=nobs,
                p=p, p_lo=p_lo, p_hi=p_hi)


def print_occupancy(res):
    print(f"\n  --- class: {res['label']} (mass>{MASS_MIN:g} M⊕; corner = insol<{COLD_MAX:g} I⊕) ---")
    print(f"  {'stratum':<8}{'D_cold':>8}{'D_hot':>8}{'C_cold':>8}{'C_hot':>8}{'ratio':>8}"
          f"{'N_hot':>7}{'λ':>8}{'±':>7}{'naive':>8}{'N_obs':>7}")
    for r in res["rows"]:
        print(f"  {r['stratum']:<8}{r['D_cold']:>8.4f}{r['D_hot']:>8.4f}{r['c_cold']:>8d}{r['c_hot']:>8d}"
              f"{r['ratio']:>8.3f}{r['n_hot']:>7d}{r['lam']:>8.2f}{r['sig']:>7.2f}{r['naive']:>8.1f}{r['n_obs']:>7d}")
    print(f"  TOTAL: λ = {res['lam']:.2f} ± {res['sig']:.2f}   naive (no selection) = {res['naive']:.1f}   "
          f"N_obs = {res['nobs']}")
    print(f"  Poisson p(X ≤ {res['nobs']} | λ={res['lam']:.2f}) = {res['p']:.4f}   "
          f"[λ−σ: {res['p_lo']:.4f} / λ+σ: {res['p_hi']:.4f}]")


def rocky_share_test(res_rocky, res_puffy):
    """Rocky share of the corner, observed vs flat+selection expectation. Follow-up
    targeting effort cancels here: rocky and puffy corner planets share stars/programs."""
    k, ntot = res_rocky["nobs"], res_rocky["nobs"] + res_puffy["nobs"]
    lam_tot = res_rocky["lam"] + res_puffy["lam"]
    if ntot == 0 or lam_tot <= 0:
        return None
    p_exp = res_rocky["lam"] / lam_tot
    bt = binomtest(k, ntot, p_exp, alternative="less")
    return dict(k=k, n=ntot, share=k / ntot, p_exp=p_exp, p=float(bt.pvalue))


def verdict(res):
    if res["lam"] < 3.0:
        return ("λ < 3: the corner is nearly UNDETECTABLE under this selection model — "
                "emptiness is consistent with selection alone; formation unconstrained here.")
    if res["p"] < 0.05 and res["nobs"] < res["lam"]:
        return (f"corner IS detectable (λ={res['lam']:.1f}) yet NASA holds {res['nobs']} "
                f"(p={res['p']:.4f}) — deficit BEYOND selection → formation/retention signal.")
    return (f"corner detectable (λ={res['lam']:.1f}), observed {res['nobs']} consistent with "
            f"the flat null (p={res['p']:.3f}) — no deficit beyond selection at this precision.")


def detection_map(flat, subset):
    """D per (insolation, radius) cell for the given flat subset; NaN where n < MIN_CELL."""
    fi = np.digitize(flat["flux"], FLUX_EDGES) - 1
    ri = np.digitize(flat["radius"], RADIUS_EDGES) - 1
    nF, nR = len(FLUX_EDGES) - 1, len(RADIUS_EDGES) - 1
    D = np.full((nR, nF), np.nan)
    N = np.zeros((nR, nF), int)
    ok = subset & (fi >= 0) & (fi < nF) & (ri >= 0) & (ri < nR)
    for r in range(nR):
        for f in range(nF):
            cell = ok & (ri == r) & (fi == f)
            n = int(cell.sum())
            N[r, f] = n
            if n >= MIN_CELL:
                D[r, f] = flat["det"][cell].mean()
    return D, N


def make_figure(flat, rocky_flat, res_rocky, res_puffy, share):
    fig, axes = plt.subplots(2, 2, figsize=(17, 11.5))
    X, Y = np.meshgrid(FLUX_EDGES, RADIUS_EDGES)
    vmax = 0.0
    maps = {}
    for name, tlo, thi in STRATA:
        sub = rocky_flat & (flat["mass"] > MASS_MIN) & (flat["teff"] >= tlo) & (flat["teff"] < thi)
        D, N = detection_map(flat, sub)
        maps[name] = (D, N)
        vmax = max(vmax, np.nanmax(D) if np.isfinite(D).any() else 0.0)

    for letter, ax, (name, _, _) in zip("ab", axes[0], STRATA):
        D, N = maps[name]
        pc = ax.pcolormesh(X, Y, D, cmap="viridis", vmin=0, vmax=vmax)
        fig.colorbar(pc, ax=ax, label="joint transit+RV detected fraction D")
        for r in range(D.shape[0]):
            for f in range(D.shape[1]):
                if not np.isfinite(D[r, f]):
                    ax.add_patch(plt.Rectangle((FLUX_EDGES[f], RADIUS_EDGES[r]),
                                               FLUX_EDGES[f + 1] - FLUX_EDGES[f],
                                               RADIUS_EDGES[r + 1] - RADIUS_EDGES[r],
                                               hatch="xx", fill=False, ec="0.6", lw=0))
        ax.axvline(COLD_MAX, color="red", ls="--", lw=1.5)
        ax.text(COLD_MAX * 0.8, RADIUS_EDGES[-1] - 0.03, "← corner (I<50)", color="red",
                fontsize=9, ha="right", va="top")
        ax.set_xscale("log")
        ax.set_xlabel(r"insolation [$I_\oplus$]"); ax.set_ylabel(r"planet radius [$R_\oplus$]")
        ax.set_title(f"({letter}) D(I, R) — rocky, mass>{MASS_MIN:g} M⊕ — {name} hosts\n"
                     f"(hatched: <{MIN_CELL} flat planets)", fontsize=10)

    # (c) expected vs observed corner counts, rocky
    ax = axes[1, 0]
    names = [r["stratum"] for r in res_rocky["rows"]] + ["TOTAL"]
    naive = [r["naive"] for r in res_rocky["rows"]] + [res_rocky["naive"]]
    lam = [r["lam"] for r in res_rocky["rows"]] + [res_rocky["lam"]]
    sig = [r["sig"] for r in res_rocky["rows"]] + [res_rocky["sig"]]
    obs = [r["n_obs"] for r in res_rocky["rows"]] + [res_rocky["nobs"]]
    x = np.arange(len(names)); w = 0.27
    ax.bar(x - w, naive, w, facecolor="none", edgecolor="tab:blue", hatch="//",
           label="flat occurrence, NO selection (naive)")
    ax.bar(x, lam, w, color="tab:blue", yerr=sig, capsize=3,
           label="expected WITH selection  λ = Σ N_hot·D_cold/D_hot")
    ax.bar(x + w, obs, w, color="tab:green", label="NASA observed")
    for xi, v in zip(x - w, naive):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x, lam):
        ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w, obs):
        ax.text(xi, v, f"{v:d}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_yscale("log"); ax.set_ylim(1, max(naive) * 8)
    ax.set_ylabel("cold-corner ROCKY (M>2) planet count")
    ax.set_title("(c) corner occupancy: expected vs observed (rocky)", fontsize=10)
    ax.grid(alpha=0.2, axis="y"); ax.legend(fontsize=7.5, loc="upper left")
    box = (f"puffy control: λ={res_puffy['lam']:.1f}±{res_puffy['sig']:.1f}, "
           f"obs={res_puffy['nobs']}, p={res_puffy['p']:.3f}")
    if share:
        box += (f"\nrocky share of corner: {share['k']}/{share['n']} = {share['share']:.2f} "
                f"vs {share['p_exp']:.2f} expected → p={share['p']:.1e}\n"
                f"(share cancels follow-up targeting; absolute counts do not)")
    ax.text(0.985, 0.975, box, transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            bbox=dict(boxstyle="round", fc="whitesmoke", ec="0.7"))

    # (d) Poisson pmf, rocky
    ax = axes[1, 1]
    lam_t, nobs = res_rocky["lam"], res_rocky["nobs"]
    ks = np.arange(0, int(max(nobs * 1.6 + 6, lam_t * 2.2 + 6)))
    pmf = poisson.pmf(ks, lam_t)
    ax.bar(ks, pmf, color="0.8", width=0.9)
    ax.bar(ks[ks <= nobs], pmf[ks <= nobs], color="tab:green", alpha=0.75, width=0.9,
           label=f"P(X ≤ {nobs}) = {res_rocky['p']:.4f}")
    ax.axvline(lam_t, color="tab:blue", ls="-", lw=2, label=f"λ = {lam_t:.2f} ± {res_rocky['sig']:.2f}")
    ax.axvspan(lam_t - res_rocky["sig"], lam_t + res_rocky["sig"], color="tab:blue", alpha=0.12)
    ax.axvline(nobs, color="tab:green", ls="--", lw=2, label=f"NASA observed = {nobs}")
    ax.set_xlabel("cold-corner rocky count k"); ax.set_ylabel("Poisson pmf")
    ax.set_title("(d) Poisson test on the corner count (rocky)", fontsize=10)
    ax.grid(alpha=0.2); ax.legend(fontsize=9)

    fig.suptitle("TEST A — completeness-corrected occupancy of the cold super-Earth corner "
                 f"(rocky, mass>{MASS_MIN:g} M⊕, insol<{COLD_MAX:g} I⊕)\n"
                 "D from the FLAT universe (no occurrence prior); absolute rate calibrated on NASA's "
                 "HOT rocky count per star stratum; Poisson on the corner count",
                 fontsize=12)
    fig.text(0.5, 0.005,
             "Caveats: D is a toy transit+RV proxy for NASA's heterogeneous selection (nearby flat stars; "
             "hot-calibrated ratio uses only the cold/hot SHAPE of D, not its absolute level); "
             "null = occurrence flat in the generator measure (log-uniform P, uniform R); "
             "counts use catalog central values (label noise negligible at region scale vs Poisson error).",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])
    out_png = os.path.join(OUT_DIR, "corner_occupancy_poisson.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    print(f"--> building flat pool (N={FLAT_N_POOL}) + transit/RV detectors...")
    flat = build_flat()
    rocky_flat = flat["radius"] < np.interp(flat["mass"], m_sil, r_sil)
    print(f"    pool in box: {len(flat['mass'])}, joint-detected: {int(flat['det'].sum())}")

    results = {}
    for precision, tag in [(False, "FULL measured-mass NASA (primary)"),
                           (True, "precision-cut NASA (sensitivity)")]:
        nasa = load_nasa(precision)
        rocky_nasa = nasa["r"] < np.interp(nasa["m"], m_sil, r_sil)
        print(f"\n  ================ TEST A — {tag} ================")
        res_rocky = occupancy("ROCKY", rocky_flat, rocky_nasa, flat, nasa)
        res_puffy = occupancy("PUFFY (control)", ~rocky_flat, ~rocky_nasa, flat, nasa)
        print_occupancy(res_rocky)
        print(f"  VERDICT: {verdict(res_rocky)}")
        print_occupancy(res_puffy)
        print(f"  control reading: {verdict(res_puffy)}")
        share = rocky_share_test(res_rocky, res_puffy)
        if share:
            print(f"\n  ROCKY SHARE of the corner (targeting-effort cancels): observed "
                  f"{share['k']}/{share['n']} = {share['share']:.2f} vs expected {share['p_exp']:.2f} "
                  f"(flat+selection); binomial p(≤obs) = {share['p']:.2e}")
        results[precision] = (res_rocky, res_puffy, share)

    res_rocky, res_puffy, share = results[False]
    make_figure(flat, rocky_flat, res_rocky, res_puffy, share)


if __name__ == "__main__":
    main()

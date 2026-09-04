"""
37_hierarchical_rocky_mixture.py — TEST E: selection-aware rocky/volatile mixture.

The rigorous deconvolution of formation from selection (Rogers 2015-style conditional
mixture; hierarchical M-R in the spirit of Wolfgang+2016 / Chen & Kipping 2017, made
selection-aware with THIS repo's calibrated RV detector as the selection function).

Model, per observed NASA planet i with (R_obs, logM_obs, sigma_logM, I, star):
  true population at fixed R = mixture of a ROCKY mass branch and a VOLATILE mass branch,
      rocky:    R = C_r * M^beta_r  ->  logM ~ N(mu_rock(R_obs),  s_rock^2)
      volatile: R = C_v * M^beta_v  ->  logM ~ N(mu_vol(R_obs),   s_vol^2)
  with mixture weight f = f_rocky per CELL (R bin x I slice).
  Selection at fixed (R, P, star): transit probability and the brightness gate are
  MASS-INDEPENDENT -> they cancel in the conditional likelihood. Only the RV mass
  threshold survives:
      eta_i(M) = Phi( (log10 M - log10 M50_i) / X_SOFT ),   M50_i = SNR_MIN*sigma_K,i/K1_i
  where K1_i = repo RVData semi-amplitude at 1 M_earth and sigma_K,i its K-precision
  (best of HARPS/NIRPS per target, matching run_rv_best). X_SOFT->0 reproduces the
  repo's hard K/sigma_K >= 5 rule exactly.
  Per-planet conditional likelihood (detected planet, observed mass):
      L_i(f) = [ f*A_i + (1-f)*B_i ] / [ f*C_i + (1-f)*D_i ]
      A_i = int rock_pdf(logM) * eta_i(M) * N(logM_obs; logM, sigma_logM) dlogM   (B_i: volatile)
      C_i = int rock_pdf(logM) * eta_i(M) dlogM                                   (D_i: volatile)
  Posterior per cell: flat Beta(1,1) prior on f, exact 1-D grid (no MCMC).

Knobs (all disclosed, all varied in the sensitivity tables):
  rocky branch     Otegi+2020 (C=1.03, beta=0.29) default; CK17/Edmondson/Mueller sensitivity
  rocky scatter    0.15 dex (script 77 MR_SCATTER_DEX)
  volatile branch  R = 0.70*M^0.63 = Otegi+2020 volatile-rich (M=1.74 R^1.58, A&A 634 A43;
                   coefficients VERIFIED). scatter 0.20 dex is an ASSUMED intrinsic dispersion
                   (not an Otegi-reported value) — bracketed by the volatile-knob sensitivity row.
  X_SOFT           0.15 dex default; 0.05 / 0.30 sensitivity
  eta on/off       off = classification-only mixture (no selection correction)

Run:
    python scripts/37_hierarchical_rocky_mixture.py
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
from scipy.stats import norm

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from run.ppop.flat_detect import RVData

SILICATE_CURVE = Path(SILICON_CURVE)
NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
OUT_DIR = os.path.join(ROOT, "output/plots", "37_hierarchical_rocky_mixture")

BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)
NASA_MASS_PREC = 0.25
NASA_RAD_PREC = 0.08
COLD_MAX = 50.0
SNR_MIN = 5.0
X_SOFT = 0.15
ROCKY_SCATTER = 0.15
VOL_C, VOL_BETA, VOL_SCATTER = 0.70, 0.63, 0.20
SIGMA_LOGM_FLOOR = 0.02
SIGMA_LOGM_MISSING = 0.20

RELATIONS = [
    ("Chen&Kipping17", 1.01, 0.28),
    ("Otegi+20 (def)", 1.03, 0.29),
    ("Edmondson+23", 0.99, 0.34),
    ("Mueller+24", 1.02, 0.27),
]
DEFAULT_REL = 1

R_EDGES = np.array([1.0, 1.4, 1.8, 2.2])
CELL_RBINS = [(1.0, 1.4), (1.4, 1.8), (1.8, 2.2), (1.4, 2.2), (1.0, 2.2)]
CELL_LABELS = ["1.0–1.4", "1.4–1.8", "1.8–2.2", "large 1.4–2.2", "pooled 1.0–2.2"]

LOGM_GRID = np.linspace(np.log10(0.05), 2.0, 261)
F_GRID = np.linspace(0.0, 1.0, 401)
EPS = 1e-300
TRAPZ = getattr(np, "trapezoid", np.trapz)


def branch_crossing(c_rock, beta_rock, c_vol=VOL_C, beta_vol=VOL_BETA):
    """Radius where the two branch mean masses coincide — below it the mixture is
    weakly identified (branches overlap; f driven by scatter differences only)."""
    logr = (beta_vol * np.log10(c_rock) - beta_rock * np.log10(c_vol)) / (beta_vol - beta_rock)
    return 10.0 ** logr


def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    o = np.argsort(m)
    return m[o], r[o]


def load_nasa(precision: bool):
    df = pd.read_csv(NASA_FILE)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    ins = pd.to_numeric(df["pl_insol"], errors="coerce")
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    meas = prov.str.contains("Mass|Msini", case=False, na=False) & \
        ~prov.str.contains("Calc", case=False, na=False)
    me = np.maximum(pd.to_numeric(df["pl_bmasseerr1"], errors="coerce").abs(),
                    pd.to_numeric(df["pl_bmasseerr2"], errors="coerce").abs())
    re = np.maximum(pd.to_numeric(df["pl_radeerr1"], errors="coerce").abs(),
                    pd.to_numeric(df["pl_radeerr2"], errors="coerce").abs())
    keep = (meas & r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"])
            & ins.between(BOX["f_lo"], BOX["f_hi"]))
    if precision:
        keep = keep & (me / m <= NASA_MASS_PREC) & (re / r <= NASA_RAD_PREC)

    teff = pd.to_numeric(df["st_teff"], errors="coerce")
    rs = pd.to_numeric(df["st_rad"], errors="coerce")
    ms = pd.to_numeric(df["st_mass"], errors="coerce")
    llog = pd.to_numeric(df["st_lum"], errors="coerce")
    dist = pd.to_numeric(df["sy_dist"], errors="coerce")

    lum = np.power(10.0, llog)
    lum_fb = rs ** 2 * (teff / 5772.0) ** 4
    n_lum_fb = int((lum.isna() & lum_fb.notna() & keep).sum())
    lum = lum.fillna(lum_fb)
    mstar = ms.copy()
    n_ms_fb = int((mstar.isna() & rs.notna() & keep).sum())
    mstar = mstar.fillna(rs)                      # lower-MS M ~ R (same rule as RVData)

    ok_star = lum.notna() & (lum > 0) & mstar.notna() & (mstar > 0) & \
        teff.notna() & dist.notna() & rs.notna()
    n_drop = int((keep & ~ok_star).sum())
    keep = keep & ok_star

    a_au = np.sqrt(lum / ins)                     # I = L / a^2  (Earth units)
    p_orb = np.sqrt(a_au ** 3 / mstar) * 365.25   # Kepler III, days

    s_logm = (me / m / np.log(10.0)).clip(lower=SIGMA_LOGM_FLOOR)
    s_logm = s_logm.fillna(SIGMA_LOGM_MISSING)
    s_logr = (re / r / np.log(10.0)).fillna(NASA_RAD_PREC / np.log(10.0)).clip(lower=0.005)

    k = keep.to_numpy(bool)
    return dict(
        m=m.to_numpy(float)[k], r=r.to_numpy(float)[k], ins=ins.to_numpy(float)[k],
        s_logm=s_logm.to_numpy(float)[k], s_logr=s_logr.to_numpy(float)[k],
        teff=teff.to_numpy(float)[k], rs=rs.to_numpy(float)[k],
        mstar=mstar.to_numpy(float)[k], lum=lum.to_numpy(float)[k],
        dist=dist.to_numpy(float)[k], p_orb=p_orb.to_numpy(float)[k],
        n=int(k.sum()), n_drop=n_drop, n_lum_fb=n_lum_fb, n_ms_fb=n_ms_fb,
    )


def rv_m50(nasa):
    """Per-planet RV 50%-mass threshold M50 = SNR_MIN*sigma_K/K1 [M_earth],
    best (smallest M50) of HARPS/NIRPS among instruments whose brightness gate passes."""
    base = pd.DataFrame(dict(
        mass_p=1.0, radius_p=nasa["r"], p_orb=nasa["p_orb"], ecc_p=0.0, inc_p=90.0,
        teff_s=nasa["teff"], radius_s=nasa["rs"], mass_s=nasa["mstar"],
        l_sun=nasa["lum"], distance_s=nasa["dist"],
    ))
    m50s, brights = [], []
    for inst in ["HARPS", "NIRPS"]:
        rv = RVData(base.copy(), source="ppop", instrument=inst)
        cat = rv.determine_detectable()
        k1 = pd.to_numeric(cat["rv_k_ms"], errors="coerce").to_numpy(float)
        sk = pd.to_numeric(cat["rv_sigma_K_ms"], errors="coerce").to_numpy(float)
        m50s.append(SNR_MIN * sk / np.clip(k1, 1e-12, None))
        brights.append(rv.bright_enough().to_numpy(bool))
    m50s, brights = np.array(m50s), np.array(brights)
    m50_masked = np.where(brights, m50s, np.inf)
    m50 = m50_masked.min(axis=0)
    n_faint = int((~brights.any(axis=0)).sum())
    m50 = np.where(np.isfinite(m50), m50, m50s.min(axis=0))   # neither bright: keep shape
    return m50, n_faint


def branch_mu_s(r_obs, s_logr, c, beta, scatter):
    mu = (np.log10(r_obs) - np.log10(c)) / beta
    s = np.sqrt(scatter ** 2 + (s_logr / beta) ** 2)
    return mu, s


def planet_integrals(nasa, m50, c_rock, beta_rock, x_soft, eta_on=True,
                     vol_c=VOL_C, vol_beta=VOL_BETA, vol_scatter=VOL_SCATTER):
    """A, B, C, D per planet (trapezoid on LOGM_GRID)."""
    g = LOGM_GRID[None, :]
    mu_r, s_r = branch_mu_s(nasa["r"], nasa["s_logr"], c_rock, beta_rock, ROCKY_SCATTER)
    mu_v, s_v = branch_mu_s(nasa["r"], nasa["s_logr"], vol_c, vol_beta, vol_scatter)
    rock = norm.pdf(g, mu_r[:, None], s_r[:, None])
    vol = norm.pdf(g, mu_v[:, None], s_v[:, None])
    if eta_on:
        eta = norm.cdf((g - np.log10(m50)[:, None]) / x_soft)
    else:
        eta = np.ones_like(rock)
    kern = norm.pdf(np.log10(nasa["m"])[:, None], g, nasa["s_logm"][:, None])
    A = TRAPZ(rock * eta * kern, LOGM_GRID, axis=1)
    B = TRAPZ(vol * eta * kern, LOGM_GRID, axis=1)
    C = TRAPZ(rock * eta, LOGM_GRID, axis=1)
    D = TRAPZ(vol * eta, LOGM_GRID, axis=1)
    return A, B, C, D


def cell_posterior(A, B, C, D, idx):
    """Normalized posterior pdf of f on F_GRID for planets idx (flat prior)."""
    if idx.sum() == 0:
        p = np.ones_like(F_GRID)
        return p / TRAPZ(p, F_GRID)
    f = F_GRID[None, :]
    num = f * A[idx][:, None] + (1 - f) * B[idx][:, None]
    den = f * C[idx][:, None] + (1 - f) * D[idx][:, None]
    logl = np.sum(np.log(np.clip(num, EPS, None)) - np.log(np.clip(den, EPS, None)), axis=0)
    logl -= logl.max()
    p = np.exp(logl)
    return p / TRAPZ(p, F_GRID)


def summarize(p):
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(F_GRID))])
    cdf /= cdf[-1]
    q = np.interp([0.025, 0.16, 0.5, 0.84, 0.975], cdf, F_GRID)
    return dict(lo95=q[0], lo68=q[1], med=q[2], hi68=q[3], hi95=q[4])


def prob_less(p_cold, p_hot):
    """P(f_cold < f_hot) on the shared grid (0.5 credit at ties)."""
    pm_c = p_cold / p_cold.sum()
    pm_h = p_hot / p_hot.sum()
    cum_c = np.cumsum(pm_c) - 0.5 * pm_c
    return float(np.sum(pm_h * cum_c))


def cell_indices(nasa):
    cells = {}
    for (lo, hi), lbl in zip(CELL_RBINS, CELL_LABELS):
        inbin = (nasa["r"] >= lo) & (nasa["r"] < hi)
        cells[(lbl, "cold")] = inbin & (nasa["ins"] < COLD_MAX)
        cells[(lbl, "hot")] = inbin & (nasa["ins"] >= COLD_MAX)
    return cells


def raw_fractions(nasa, m_sil, r_sil):
    rocky = nasa["r"] < np.interp(nasa["m"], m_sil, r_sil)
    out = {}
    for key, idx in cell_indices(nasa).items():
        n = int(idx.sum())
        out[key] = (int(rocky[idx].sum()), n, rocky[idx].sum() / n if n else np.nan)
    return out


def run_config(nasa, m50, c_rock, beta_rock, x_soft, eta_on=True,
               vol_c=VOL_C, vol_scatter=VOL_SCATTER):
    A, B, C, D = planet_integrals(nasa, m50, c_rock, beta_rock, x_soft, eta_on,
                                  vol_c=vol_c, vol_scatter=vol_scatter)
    cells = cell_indices(nasa)
    post = {k: cell_posterior(A, B, C, D, idx) for k, idx in cells.items()}
    stats = {k: summarize(p) for k, p in post.items()}
    n_cell = {k: int(idx.sum()) for k, idx in cells.items()}
    sel_ratio = {k: float(np.median((C[idx] / np.clip(D[idx], EPS, None))))
                 if idx.sum() else np.nan for k, idx in cells.items()}
    return dict(post=post, stats=stats, n=n_cell, sel_ratio=sel_ratio)


def print_main_table(tag, res_on, res_off, raw):
    print(f"\n  ================ TEST E — {tag} ================")
    hdr = (f"  {'cell':<16}{'slice':<6}{'N':>4}  {'raw k/n':>9}  "
           f"{'f_rocky med [68%]':>22}  {'eta OFF med':>12}  {'med C/D':>9}")
    print(hdr)
    for lbl in CELL_LABELS:
        for s in ["cold", "hot"]:
            k = (lbl, s)
            st, so = res_on["stats"][k], res_off["stats"][k]
            kk, nn, fr = raw[k]
            rawtxt = f"{kk}/{nn}" if nn else "--"
            print(f"  {lbl:<16}{s:<6}{res_on['n'][k]:>4}  {rawtxt:>9}  "
                  f"{st['med']:.2f} [{st['lo68']:.2f},{st['hi68']:.2f}]".rjust(22) + "  "
                  f"{so['med']:.2f}".rjust(12) + "  "
                  f"{res_on['sel_ratio'][k]:.2f}".rjust(9))
    print(f"\n  {'cell':<16}{'P(f_cold < f_hot)':>19}  {'eta OFF':>9}")
    for lbl in CELL_LABELS:
        p_on = prob_less(res_on["post"][(lbl, "cold")], res_on["post"][(lbl, "hot")])
        p_off = prob_less(res_off["post"][(lbl, "cold")], res_off["post"][(lbl, "hot")])
        print(f"  {lbl:<16}{p_on:>19.3f}  {p_off:>9.3f}")
    key = ("large 1.4–2.2", "cold")
    print(f"\n  HEADLINE: cold large (R 1.4–2.2, I<{COLD_MAX:g}) f_rocky 95% upper bound = "
          f"{res_on['stats'][key]['hi95']:.2f} (selection-corrected; "
          f"{res_off['stats'][key]['hi95']:.2f} without selection), N={res_on['n'][key]}")


def print_sensitivity(nasa, m50):
    lbl = "large 1.4–2.2"
    print("\n  ---- sensitivity: rocky M-R relation (X_SOFT=0.15, eta on; cell = large 1.4–2.2) ----")
    print(f"  {'relation':<18}{'f_cold med [68%]':>20}  {'f_hot med [68%]':>20}  {'P(f_c<f_h)':>11}")
    for name, c, b in RELATIONS:
        res = run_config(nasa, m50, c, b, X_SOFT)
        sc, sh = res["stats"][(lbl, "cold")], res["stats"][(lbl, "hot")]
        pl = prob_less(res["post"][(lbl, "cold")], res["post"][(lbl, "hot")])
        print(f"  {name:<18}"
              + f"{sc['med']:.2f} [{sc['lo68']:.2f},{sc['hi68']:.2f}]".rjust(20) + "  "
              + f"{sh['med']:.2f} [{sh['lo68']:.2f},{sh['hi68']:.2f}]".rjust(20) + "  "
              + f"{pl:.3f}".rjust(11))
    print("\n  ---- sensitivity: selection softness X_SOFT (Otegi rocky; cell = large 1.4–2.2) ----")
    print(f"  {'X_SOFT [dex]':<18}{'f_cold med [68%]':>20}  {'f_hot med [68%]':>20}  {'P(f_c<f_h)':>11}")
    _, c, b = RELATIONS[DEFAULT_REL]
    for xs in [0.05, 0.15, 0.30]:
        res = run_config(nasa, m50, c, b, xs)
        sc, sh = res["stats"][(lbl, "cold")], res["stats"][(lbl, "hot")]
        pl = prob_less(res["post"][(lbl, "cold")], res["post"][(lbl, "hot")])
        print(f"  {xs:<18.2f}"
              + f"{sc['med']:.2f} [{sc['lo68']:.2f},{sc['hi68']:.2f}]".rjust(20) + "  "
              + f"{sh['med']:.2f} [{sh['lo68']:.2f},{sh['hi68']:.2f}]".rjust(20) + "  "
              + f"{pl:.3f}".rjust(11))
    print("\n  ---- sensitivity: VOLATILE branch knobs (pure knob shifts, NOT literature values; "
          "cell = large 1.4–2.2) ----")
    print(f"  {'volatile knob':<28}{'f_cold med [68%]':>20}  {'f_hot med [68%]':>20}  {'P(f_c<f_h)':>11}")
    for kn, vc, vs in [("scatter 0.15 dex", VOL_C, 0.15), ("scatter 0.20 (def)", VOL_C, 0.20),
                       ("scatter 0.30 dex", VOL_C, 0.30), ("C_v x 0.90", 0.9 * VOL_C, VOL_SCATTER),
                       ("C_v x 1.10", 1.1 * VOL_C, VOL_SCATTER)]:
        res = run_config(nasa, m50, c, b, X_SOFT, vol_c=vc, vol_scatter=vs)
        sc, sh = res["stats"][(lbl, "cold")], res["stats"][(lbl, "hot")]
        pl = prob_less(res["post"][(lbl, "cold")], res["post"][(lbl, "hot")])
        print(f"  {kn:<28}"
              + f"{sc['med']:.2f} [{sc['lo68']:.2f},{sc['hi68']:.2f}]".rjust(20) + "  "
              + f"{sh['med']:.2f} [{sh['lo68']:.2f},{sh['hi68']:.2f}]".rjust(20) + "  "
              + f"{pl:.3f}".rjust(11))


def make_figure(res_on, res_off, raw):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.6))
    for j, (ax, lbl) in enumerate(zip(axes[:3], CELL_LABELS[:3])):
        for s, col in [("cold", "tab:blue"), ("hot", "tab:red")]:
            p = res_on["post"][(lbl, s)]
            ax.fill_between(F_GRID, 0, p, color=col, alpha=0.15)
            ax.plot(F_GRID, p, color=col, lw=2,
                    label=f"{s} (N={res_on['n'][(lbl, s)]})" if j == 0 else None)
            ax.plot(F_GRID, res_off["post"][(lbl, s)], color=col, lw=1.2, ls="--",
                    label=f"{s}, no selection" if j == 0 else None)
        pl = prob_less(res_on["post"][(lbl, "cold")], res_on["post"][(lbl, "hot")])
        ax.text(0.03, 0.95, f"P(f_cold < f_hot) = {pl:.3f}", transform=ax.transAxes,
                fontsize=9, va="top",
                bbox=dict(boxstyle="round", fc="whitesmoke", ec="0.7"))
        if j > 0:
            ax.text(0.03, 0.82,
                    f"cold N={res_on['n'][(lbl, 'cold')]}, hot N={res_on['n'][(lbl, 'hot')]}",
                    transform=ax.transAxes, fontsize=8.5, va="top", color="0.3")
        ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
        ax.set_xlabel("f_rocky"); ax.set_ylabel("posterior density")
        ax.set_title(f"({'abc'[j]}) R {lbl} $R_\\oplus$", fontsize=11)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8.5, loc="center left")

    ax = axes[3]
    xs = np.arange(3)
    for s, col, dx, mk in [("cold", "tab:blue", -0.10, "o"), ("hot", "tab:red", 0.10, "s")]:
        med = [res_on["stats"][(l, s)]["med"] for l in CELL_LABELS[:3]]
        lo = [res_on["stats"][(l, s)]["lo68"] for l in CELL_LABELS[:3]]
        hi = [res_on["stats"][(l, s)]["hi68"] for l in CELL_LABELS[:3]]
        ax.errorbar(xs + dx, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)],
                    fmt=mk, color=col, ms=7, capsize=4, lw=1.6, label=f"{s} (selection-corrected)")
        med_off = [res_off["stats"][(l, s)]["med"] for l in CELL_LABELS[:3]]
        ax.plot(xs + dx + 0.05, med_off, mk, mfc="none", mec=col, ms=7, mew=1.4,
                label=f"{s}, no selection")
        fr = [raw[(l, s)][2] for l in CELL_LABELS[:3]]
        ax.plot(xs + dx - 0.05, fr, "x", color="0.4", ms=6, mew=1.4,
                label="raw silicate-line fraction" if s == "cold" else None)
    ax.set_xticks(xs); ax.set_xticklabels(CELL_LABELS[:3])
    ax.set_ylim(-0.04, 1.09)
    ax.set_xlabel(r"planet radius bin [$R_\oplus$]"); ax.set_ylabel("f_rocky (median, 68% CI)")
    ax.set_title("(d) summary: posterior medians", fontsize=11)
    ax.grid(alpha=0.2); ax.legend(fontsize=7.5, loc="lower left")

    fig.suptitle("TEST E — hierarchical rocky/volatile mixture with the repo RV detector as selection function\n"
                 "f_rocky per (R bin × I slice), conditional likelihood (transit + brightness cancel; "
                 "RV mass threshold survives); flat prior, exact grid posterior — "
                 "rocky = Otegi+20 (0.15 dex), volatile = 0.70·M^0.63 (0.20 dex), X_SOFT = 0.15 dex",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_png = os.path.join(OUT_DIR, "hierarchical_rocky_mixture.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    _, c_rock, beta_rock = RELATIONS[DEFAULT_REL]
    r_x = branch_crossing(c_rock, beta_rock)
    print(f"--> branch-mean crossing radius: {r_x:.2f} R_earth "
          f"(rocky and volatile mean masses coincide there; below it the mixture is "
          f"weakly identified -> treat the {R_EDGES[0]:.1f}-{R_EDGES[1]:.1f} bin as decoration)")

    for precision, tag in [(True, "precision-cut NASA (primary)"),
                           (False, "FULL measured-mass NASA (sensitivity)")]:
        nasa = load_nasa(precision)
        m50, n_faint = rv_m50(nasa)
        print(f"\n--> {tag}: N={nasa['n']} (dropped {nasa['n_drop']} missing-stellar; "
              f"lum fallback {nasa['n_lum_fb']}, mstar fallback {nasa['n_ms_fb']}, "
              f"faint-in-both-bands {n_faint})")
        print(f"    M50 [M_earth] percentiles 10/50/90: "
              f"{np.percentile(m50, 10):.2f} / {np.percentile(m50, 50):.2f} / "
              f"{np.percentile(m50, 90):.2f}")
        below = nasa["m"] < m50
        cold = nasa["ins"] < COLD_MAX
        print(f"    toy-detector mismatch (obs mass BELOW own M50 — 'should not have been "
              f"RV-detected'): {below.mean() * 100:.0f}% overall, "
              f"{below[cold].mean() * 100:.0f}% cold, {below[~cold].mean() * 100:.0f}% hot"
              + ("  [LOW -> eta trustworthy]" if below.mean() < 0.25 else
                 "  [HIGH -> eta mis-scaled for this sample; prefer eta-OFF column]"))
        res_on = run_config(nasa, m50, c_rock, beta_rock, X_SOFT, eta_on=True)
        res_off = run_config(nasa, m50, c_rock, beta_rock, X_SOFT, eta_on=False)
        raw = raw_fractions(nasa, m_sil, r_sil)
        print_main_table(tag, res_on, res_off, raw)
        if precision:
            print_sensitivity(nasa, m50)
            make_figure(res_on, res_off, raw)

    print("\n  CAVEATS (repeat with any use of this result):")
    print("  - volatile branch R=0.70*M^0.63 = Otegi+2020 volatile-rich (M=1.74 R^1.58, A&A 634")
    print("    A43; coefficients verified). The 0.20-dex intrinsic scatter is an ASSUMED value,")
    print("    not Otegi-reported — bracketed by the volatile-knob sensitivity row.")
    print("  - eta uses THIS repo's toy RVData (HARPS/NIRPS, n_obs=100, jitter_red_frac=0) as a")
    print("    proxy for the heterogeneous real follow-up; only its mass-SHAPE enters, but that")
    print("    shape is a model. X_SOFT sensitivity row is the honesty check.")
    print("  - masses include msini rows (transiting -> sin i ~ 1, bias < few %).")
    print("  - conditional-on-detection likelihood: occurrence and targeting cancel per planet;")
    print("    the model measures COMPOSITION mix among detected planets, corrected for the")
    print("    mass-selection within each cell — not absolute occurrence (that is TEST A's job).")
    print("  - insolation slicing on central pl_insol; slice-assignment noise not propagated.")
    print("  - branch means cross at ~1.4 R_earth: the 1.0-1.4 bin cannot separate the classes")
    print("    by mass; only bins above the crossing carry real mixture information.")
    print("  - where the toy-mismatch fraction is HIGH (full sample), the eta-on correction")
    print("    over-trusts M50 scales far from the real programs' sensitivity; quote eta-OFF.")


if __name__ == "__main__":
    main()

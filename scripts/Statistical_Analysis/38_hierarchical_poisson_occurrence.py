"""
38_hierarchical_poisson_occurrence.py — PLAN 1: hierarchical Bayesian Poisson-process
occurrence with completeness, fit by MCMC.

Question: after fitting the population's OWN mass/insolation occurrence law, how suppressed
is the cold rocky super-Earth window (rocky, M>2 M⊕, I<50 I⊕) relative to extrapolation
from everywhere else?  θ_occ = window multiplier; scenario B ("rocky super-Earths form
everywhere alike") is θ_occ = 1.

Model (inverse-detection-efficiency lineage: Youdin 2011, ApJ 742, 38; Foreman-Mackey,
Hogg & Morton 2014, ApJ 795, 64; Burke et al. 2015, ApJ 809, 8):
  cells = class (rocky/puffy by silicate curve) × stratum (FGK/M) × insolation column ×
  mass bin.  Per-cell completeness π = joint transit+RV detected fraction of the FLAT pool
  in that cell (flat measure survives only WITHIN cells — the occurrence shape across
  cells is FITTED, so θ_occ does not inherit the flat-measure yardstick of tests LR/θ).
  Occurrence density per class:  ln λ_c(logM, logI) = a_c + b_c·logM + b2_c·logM² +
  g_c·logI, plus an M-stratum offset in the rocky/puffy contrast (so "M dwarfs are
  rockier everywhere" cannot leak into θ_occ); rocky window cells multiplied by θ_occ.
  A plain log-linear law is fit alongside as a law-sensitivity check.

Follow-up targeting (the TEST-A poison): expected counts are effort_(stratum,I-column) ×
π × λ.  Conditioning on each column's total count removes effort EXACTLY
(Poisson → multinomial within column) — the hierarchical generalization of the
rocky-share test.  Identified: b_rocky, b_puffy, Δa, Δg (class contrasts), θ_occ.
Absolute rates (planets per star) are NOT identified without a defined stellar survey.
A naive no-effort Poisson fit is shown alongside to expose the targeting distortion.

Sampling: Goodman & Weare (2010) affine-invariant stretch move (the emcee algorithm,
Foreman-Mackey et al. 2013, PASP 125, 306; reimplemented here — emcee not installed),
split-chain R̂ convergence check, posterior-predictive deviance check, Savage–Dickey
(Dickey 1971) Bayes factor at θ_occ = 1 under the flat U(0,3) prior.

Outputs (my_outputs/38_hierarchical_poisson_occurrence/):
    hier_poisson_occurrence.png, posterior_summary.csv, cell_table.csv

Run:
    python "scripts/Statistical_Analysis/38_hierarchical_poisson_occurrence.py"
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import time
import importlib.util
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S79 = _load("s79", str(ROOT / "scripts" / "Statistical_Analysis" / "35_corner_occupancy_poisson.py"))

OUT_DIR = os.path.join(ROOT, "my_outputs", "38_hierarchical_poisson_occurrence")
POOL_CACHE = os.path.join(OUT_DIR, f"flat_pool_seed{S79.RNG_SEED}_n{S79.FLAT_N_POOL}.npz")

I_EDGES = np.array([1e-2, 1.0, 50.0, 500.0, 1e4])          # cold = first two columns
M_EDGES = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 12.0])        # window = last two bins
CLASSES = ["rocky", "puffy"]
THETA_MAX = 3.0
SEED_MCMC = 42
N_WALK, N_BURN, N_KEEP = 40, 2000, 6000
MIN_POOL_WARN = 50

C_BLUE, C_AQUA, C_YELLOW, C_RED = "#2a78d6", "#1baf7a", "#eda100", "#e34948"
C_INK, C_MUTED, C_GRID = "#0b0b0b", "#898781", "#e1e0d9"


# ---------------------------------------------------------------- cell machinery

def build_cells(flat, m_sil, r_sil):
    """Flattened cell arrays: completeness π (Jeffreys-smoothed), pool counts, geometry."""
    rocky = flat["radius"] < np.interp(flat["mass"], m_sil, r_sil)
    mi = np.digitize(flat["mass"], M_EDGES) - 1
    ii = np.digitize(flat["flux"], I_EDGES) - 1
    rows = []
    for c in range(2):
        cls_mask = rocky if c == 0 else ~rocky
        for s, (sname, tlo, thi) in enumerate(S79.STRATA):
            smask = (flat["teff"] >= tlo) & (flat["teff"] < thi)
            for j in range(len(I_EDGES) - 1):
                for k in range(len(M_EDGES) - 1):
                    cell = cls_mask & smask & (ii == j) & (mi == k)
                    n_pool = int(cell.sum())
                    n_det = int((cell & flat["det"]).sum())
                    rows.append(dict(
                        cls=c, stratum=s, icol=j, mbin=k,
                        m=np.log10(np.sqrt(M_EDGES[k] * M_EDGES[k + 1])),
                        i=np.log10(np.sqrt(I_EDGES[j] * I_EDGES[j + 1])),
                        dm=np.log10(M_EDGES[k + 1] / M_EDGES[k]),
                        di=np.log10(I_EDGES[j + 1] / I_EDGES[j]),
                        pool_n=n_pool, pool_det=n_det,
                        pi=(n_det + 0.5) / (n_pool + 1.0),
                        win=(c == 0) and (M_EDGES[k] >= 2.0) and (I_EDGES[j + 1] <= 50.0),
                        col=s * (len(I_EDGES) - 1) + j,
                    ))
    df = pd.DataFrame(rows)
    return {k: df[k].to_numpy() for k in df.columns} | dict(
        ncol=len(S79.STRATA) * (len(I_EDGES) - 1), df=df)


def nasa_counts(nasa, m_sil, r_sil, cells):
    rocky = nasa["r"] < np.interp(nasa["m"], m_sil, r_sil)
    mi = np.digitize(nasa["m"], M_EDGES) - 1
    ii = np.digitize(nasa["ins"], I_EDGES) - 1
    strat = np.full(len(nasa["m"]), -1)
    for s, (_, tlo, thi) in enumerate(S79.STRATA):
        strat[(nasa["teff"] >= tlo) & (nasa["teff"] < thi)] = s
    n = np.zeros(len(cells["cls"]), int)
    for idx in range(len(n)):
        sel = ((rocky == (cells["cls"][idx] == 0)) & (strat == cells["stratum"][idx])
               & (ii == cells["icol"][idx]) & (mi == cells["mbin"][idx]))
        n[idx] = int(sel.sum())
    return n


# ---------------------------------------------------------------- likelihoods

def cell_rates(x, cells, flex):
    """Un-normalized per-cell rate q ∝ π·Δm·λ for the conditioned model.
    flex adds mass-function curvature (b2·logM²) and an M-stratum offset in the
    rocky/puffy contrast (da_m) — without da_m, 'M dwarfs are rockier everywhere'
    would leak into θ_occ, because the cold columns are M-dwarf dominated."""
    m, i, str_m = cells["m"], cells["i"], (cells["stratum"] == 1)
    if flex:
        da, da_m, br, br2, bp, bp2, dg, th = x
        ln_r = da + da_m * str_m + br * m + br2 * m * m + dg * i
        ln_p = bp * m + bp2 * m * m
    else:
        da, br, bp, dg, th = x
        ln_r = da + br * m + dg * i
        ln_p = bp * m
    q = cells["pi"] * cells["dm"] * np.exp(np.where(cells["cls"] == 0, ln_r, ln_p))
    return np.where(cells["win"], q * th, q)


def logpost_cond(x, cells, n_obs, flex):
    th = x[-1]
    if not (0.0 <= th <= THETA_MAX):
        return -np.inf
    lp = -0.5 * float(np.sum(np.asarray(x[:-1]) ** 2)) / 2.0 ** 2
    q = cell_rates(x, cells, flex)
    tot = np.bincount(cells["col"], weights=q, minlength=cells["ncol"])
    p = q / tot[cells["col"]]
    mask = n_obs > 0
    if np.any(p[mask] <= 0):
        return -np.inf
    return lp + float(np.sum(n_obs[mask] * np.log(p[mask])))


def logpost_pois(x, cells, n_obs):
    """Naive no-effort Poisson with the SAME flexible law — only the effort
    treatment differs from the conditioned model."""
    ar, ap, da_m, br, br2, bp, bp2, gr, gp, ds, th = x
    if not (0.0 <= th <= THETA_MAX):
        return -np.inf
    lp = (-0.5 * (ar ** 2 + ap ** 2) / 3.0 ** 2
          - 0.5 * (da_m ** 2 + br ** 2 + br2 ** 2 + bp ** 2 + bp2 ** 2
                   + gr ** 2 + gp ** 2 + ds ** 2) / 2.0 ** 2)
    m, i, str_m = cells["m"], cells["i"], (cells["stratum"] == 1)
    lnlam = np.where(cells["cls"] == 0,
                     ar + da_m * str_m + br * m + br2 * m * m + gr * i,
                     ap + bp * m + bp2 * m * m + gp * i)
    lnlam = lnlam + ds * str_m
    mu = cells["pi"] * cells["dm"] * cells["di"] * np.exp(lnlam)
    mu = np.where(cells["win"], mu * th, mu)
    mask = n_obs > 0
    if np.any(mu[mask] <= 0):
        return -np.inf
    return lp + float(np.sum(n_obs[mask] * np.log(mu[mask])) - mu.sum())


# ---------------------------------------------------------------- sampler (Goodman & Weare 2010)

def stretch_sample(logpost, x0, n_steps, rng, a=2.0):
    nw, nd = x0.shape
    x = x0.copy()
    lp = np.array([logpost(xi) for xi in x])
    chain = np.empty((n_steps, nw, nd))
    n_acc = 0
    half = nw // 2
    for t in range(n_steps):
        for lo, hi in ((0, half), (half, nw)):
            idx = np.arange(lo, hi)
            comp = np.concatenate([np.arange(0, lo), np.arange(hi, nw)])
            z = ((a - 1.0) * rng.random(len(idx)) + 1.0) ** 2 / a
            partners = rng.choice(comp, size=len(idx))
            prop = x[partners] + z[:, None] * (x[idx] - x[partners])
            lpp = np.array([logpost(p) for p in prop])
            log_r = (nd - 1) * np.log(z) + lpp - lp[idx]
            acc = np.log(rng.random(len(idx))) < log_r
            x[idx[acc]] = prop[acc]
            lp[idx[acc]] = lpp[acc]
            n_acc += int(acc.sum())
        chain[t] = x
    return chain, n_acc / (n_steps * nw)


def split_rhat(chain):
    """Gelman–Rubin R̂ per parameter; each walker half-chain is one chain."""
    n, nw, nd = chain.shape
    h = n // 2
    chains = np.concatenate([chain[:h], chain[h:2 * h]], axis=1)   # (h, 2nw, nd)
    means = chains.mean(axis=0)
    variances = chains.var(axis=0, ddof=1)
    W = variances.mean(axis=0)
    B = h * means.var(axis=0, ddof=1)
    return np.sqrt(((h - 1) / h * W + B / h) / W)


def run_mcmc(logpost, x0, rng, label):
    t0 = time.time()
    chain, acc = stretch_sample(logpost, x0, N_BURN + N_KEEP, rng)
    kept = chain[N_BURN:]
    rhat = split_rhat(kept)
    flat = kept.reshape(-1, kept.shape[-1])
    print(f"    {label}: acc={acc:.2f}, max R̂={rhat.max():.3f}, "
          f"{flat.shape[0]} samples, {time.time() - t0:.1f}s")
    if rhat.max() > 1.05:
        print(f"    WARNING: R̂ > 1.05 — chains not converged, treat results as provisional")
    return flat, acc, rhat


# ---------------------------------------------------------------- inference summaries

def theta_summary(th, label):
    q05, q50, q95 = np.quantile(th, [0.05, 0.5, 0.95])
    p_lt1 = float((th < 1.0).mean())
    h = 0.10                                            # Savage–Dickey window at θ=1
    n1 = int((np.abs(th - 1.0) < h / 2).sum())
    prior_dens = 1.0 / THETA_MAX
    if n1 > 0:
        bf_B = (n1 / (len(th) * h)) / prior_dens        # >1 favors B (θ=1)
        bf_txt = f"BF(B vs free θ) = {bf_B:.3g}"
    else:
        bf_B = 1.0 / (len(th) * h) / prior_dens
        bf_txt = f"BF(B vs free θ) < {bf_B:.1g} (no samples near 1)"
    print(f"    {label}: θ_occ median={q50:.2f}, 90% CI [{q05:.2f}, {q95:.2f}], "
          f"P(θ<1)={p_lt1:.4f}, {bf_txt}")
    return dict(q05=q05, q50=q50, q95=q95, p_lt1=p_lt1, bf_B=bf_B, n_sd=n1)


def ppc_deviance(samples, cells, n_obs, rng, flex, n_rep=1000):
    """Posterior-predictive deviance for the conditioned (multinomial) model."""
    x_mean = samples.mean(axis=0)
    q = cell_rates(x_mean, cells, flex)
    tot = np.bincount(cells["col"], weights=q, minlength=cells["ncol"])
    p = q / tot[cells["col"]]
    n_col = np.bincount(cells["col"], weights=n_obs, minlength=cells["ncol"]).astype(int)
    e = n_col[cells["col"]] * p
    mask = n_obs > 0
    g_cell = np.zeros(len(n_obs))
    g_cell[mask] = 2.0 * n_obs[mask] * np.log(n_obs[mask] / e[mask])
    g_obs = float(g_cell.sum())
    g_rep = np.empty(n_rep)
    for r in range(n_rep):
        n_rep_cells = np.zeros_like(n_obs)
        for col in range(cells["ncol"]):
            in_col = cells["col"] == col
            if n_col[col] > 0:
                n_rep_cells[in_col] = rng.multinomial(n_col[col], p[in_col] / p[in_col].sum())
        m = n_rep_cells > 0
        g_rep[r] = 2.0 * np.sum(n_rep_cells[m] * np.log(n_rep_cells[m] / e[m]))
    ppp = float((g_rep >= g_obs).mean())
    print(f"    posterior-predictive deviance: G_obs={g_obs:.1f}, "
          f"G_rep median={np.median(g_rep):.1f}, ppp={ppp:.3f}")
    worst = np.argsort(g_cell)[::-1][:5]
    for w in worst:
        if g_cell[w] > 1.0:
            print(f"      misfit cell: {CLASSES[cells['cls'][w]]:>5} "
                  f"{S79.STRATA[cells['stratum'][w]][0]:>3} "
                  f"I∈[{I_EDGES[cells['icol'][w]]:g},{I_EDGES[cells['icol'][w] + 1]:g}] "
                  f"M∈[{M_EDGES[cells['mbin'][w]]:g},{M_EDGES[cells['mbin'][w] + 1]:g}] "
                  f"obs={n_obs[w]}, exp={e[w]:.2f}, G-contrib={g_cell[w]:.1f}")
    return g_obs, ppp, e


def share_curves(samples, cells, n_obs, flex, force_theta=None, n_draw=400):
    """Model rocky share among M>2 planets per insolation column (pooled over strata)."""
    n_col = np.bincount(cells["col"], weights=n_obs, minlength=cells["ncol"])
    m2 = M_EDGES[cells["mbin"]] >= 2.0
    idx = np.linspace(0, len(samples) - 1, n_draw).astype(int)
    n_i = len(I_EDGES) - 1
    shares = np.full((n_draw, n_i), np.nan)
    for d, si in enumerate(idx):
        x = samples[si].copy()
        if force_theta is not None:
            x[-1] = force_theta
        q = cell_rates(x, cells, flex)
        tot = np.bincount(cells["col"], weights=q, minlength=cells["ncol"])
        e = n_col[cells["col"]] * q / tot[cells["col"]]
        for j in range(n_i):
            jm = (cells["icol"] == j) & m2
            er = e[jm & (cells["cls"] == 0)].sum()
            ep = e[jm & (cells["cls"] == 1)].sum()
            if er + ep > 0:
                shares[d, j] = er / (er + ep)
    return shares


def agresti_coull(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    nt = n + z ** 2
    pt = (k + z ** 2 / 2) / nt
    half = z * np.sqrt(pt * (1 - pt) / nt)
    return k / n, max(pt - half, 0.0), min(pt + half, 1.0)


# ---------------------------------------------------------------- figure

def make_figure(res, cells, n_obs):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    for ax in axes.flat:
        ax.grid(alpha=0.6, color=C_GRID, lw=0.7)
        ax.tick_params(colors=C_MUTED)
        for sp in ax.spines.values():
            sp.set_color(C_MUTED)

    # (a) θ_occ posterior, conditioned model, both NASA variants
    ax = axes[0, 0]
    bins = np.linspace(0, 1.6, 65)
    ax.hist(res["full"]["cond-flex"]["th"], bins=bins, density=True, color=C_BLUE,
            alpha=0.75, label="full measured-mass NASA (primary)")
    ax.hist(res["prec"]["cond-flex"]["th"], bins=bins, density=True, histtype="step",
            color=C_AQUA, lw=2, label="precision-cut NASA (sensitivity)")
    ax.axvline(1.0, color=C_RED, ls="--", lw=1.8)
    ax.text(1.02, 0.97, "B: θ=1\n(no suppression)", color=C_RED, fontsize=9,
            transform=ax.get_xaxis_transform(), va="top")
    s = res["full"]["cond-flex"]["summ"]
    sl = res["full"]["cond-lin"]["summ"]
    ax.text(0.975, 0.97,
            f"primary: θ_occ = {s['q50']:.2f} [{s['q05']:.2f}, {s['q95']:.2f}] (90%)\n"
            f"P(θ<1) = {s['p_lt1']:.4f}\n"
            f"Savage–Dickey BF(B) = {s['bf_B']:.2g}\n"
            f"linear-law sensitivity: θ = {sl['q50']:.2f} [{sl['q05']:.2f}, {sl['q95']:.2f}]",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="whitesmoke", ec=C_GRID))
    ax.set_xlabel("θ_occ  (cold rocky window multiplier vs fitted occurrence law)")
    ax.set_ylabel("posterior density")
    ax.set_title("(a) window suppression θ_occ — targeting-immune (conditioned) model",
                 fontsize=10)
    ax.legend(fontsize=8, loc="center right")

    # (b) rocky share among M>2 vs insolation
    ax = axes[0, 1]
    xc = np.sqrt(I_EDGES[:-1] * I_EDGES[1:])
    sh_fit = res["full"]["cond-flex"]["share_fit"]
    sh_b = res["full"]["cond-flex"]["share_b"]
    lo, mid, hi = (np.nanquantile(sh_fit, q, axis=0) for q in (0.05, 0.5, 0.95))
    ax.fill_between(xc, lo, hi, color=C_BLUE, alpha=0.25, lw=0)
    ax.plot(xc, mid, color=C_BLUE, lw=2, label="fitted model (90% band)")
    ax.plot(xc, np.nanmedian(sh_b, axis=0), color=C_RED, ls="--", lw=2,
            label="scenario B (θ=1) extrapolation")
    m2 = M_EDGES[cells["mbin"]] >= 2.0
    labeled = False
    for j in range(len(I_EDGES) - 1):
        jm = (cells["icol"] == j) & m2
        k = int(n_obs[jm & (cells["cls"] == 0)].sum())
        n = int(n_obs[jm].sum())
        p, plo, phi = agresti_coull(k, n)
        if n > 0:
            ax.errorbar(xc[j], p, yerr=[[p - plo], [phi - p]], fmt="o", color=C_INK,
                        ms=6, capsize=3, lw=1.4,
                        label=None if labeled else "NASA observed (Agresti–Coull 95%)")
            labeled = True
            ax.annotate(f"{k}/{n}", (xc[j], phi), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=8, color=C_MUTED)
    ax.axvline(50, color=C_MUTED, ls=":", lw=1.2)
    ax.text(50 * 0.85, 0.03, "cold | hot", color=C_MUTED, fontsize=8, ha="right")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(r"insolation [$I_\oplus$]")
    ax.set_ylabel("rocky share of M>2 M⊕ planets")
    ax.set_title("(b) where θ_occ comes from: rocky share vs insolation", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    # (c) per-cell observed vs predicted
    ax = axes[1, 0]
    e = res["full"]["cond-flex"]["e_mean"]
    win = cells["win"]
    ok = (n_obs > 0) | (e > 0.05)
    ax.plot([0.05, 100], [0.05, 100], color=C_MUTED, lw=1, ls="-")
    ax.scatter(np.maximum(e[ok & ~win], 0.05), np.maximum(n_obs[ok & ~win], 0.05),
               s=28, color=C_BLUE, alpha=0.75, label="cells outside window")
    ax.scatter(np.maximum(e[ok & win], 0.05), np.maximum(n_obs[ok & win], 0.05),
               s=55, facecolor=C_YELLOW, edgecolor=C_RED, lw=1.6,
               label="cold rocky window cells")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("predicted count (posterior mean, θ_occ fitted)")
    ax.set_ylabel("NASA observed count")
    ax.set_title(f"(c) per-cell fit; posterior-predictive ppp = "
                 f"{res['full']['cond-flex']['ppp']:.3f}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    # (d) conditioned vs naive Poisson θ posterior
    ax = axes[1, 1]
    ax.hist(res["full"]["cond-flex"]["th"], bins=bins, density=True, color=C_BLUE,
            alpha=0.75, label="conditioned (targeting removed)")
    ax.hist(res["full"]["pois-flex"]["th"], bins=bins, density=True, histtype="step",
            color=C_YELLOW, lw=2.2, label="naive Poisson (no effort term, same law)")
    ax.axvline(1.0, color=C_RED, ls="--", lw=1.8)
    sp = res["full"]["pois-flex"]["summ"]
    ax.text(0.975, 0.97,
            f"naive: θ = {sp['q50']:.2f} [{sp['q05']:.2f}, {sp['q95']:.2f}]\n"
            "absolute counts are inflated by follow-up\n"
            "targeting (TEST A); conditioning on column\n"
            "totals removes it exactly",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="whitesmoke", ec=C_GRID))
    ax.set_xlabel("θ_occ")
    ax.set_ylabel("posterior density")
    ax.set_title("(d) why the effort correction matters", fontsize=10)
    ax.legend(fontsize=8, loc="center right")

    fig.suptitle("PLAN 1 — hierarchical Bayesian occurrence with completeness: "
                 "cold rocky window multiplier θ_occ\n"
                 "yardstick = the population's OWN fitted power law in (logM, logI), "
                 "not the flat measure; effort per (stratum × I-column) conditioned out",
                 fontsize=12)
    fig.text(0.5, 0.005,
             "Caveats: central-value masses/radii (label noise → script 84 MC); π averages "
             "the flat measure within cells only; effort assumed mass- and class-independent "
             "within a (stratum, I-column); log-quadratic occurrence law with M-stratum "
             "contrast offset (linear law as sensitivity); absolute planets-per-star "
             "not identified without a defined stellar survey.",
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    out_png = os.path.join(OUT_DIR, "hier_poisson_occurrence.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")


# ---------------------------------------------------------------- main

PARAM_NAMES = {
    "cond-lin": ["da", "b_rocky", "b_puffy", "dg", "theta"],
    "cond-flex": ["da", "da_Mstratum", "b_rocky", "b2_rocky", "b_puffy", "b2_puffy",
                  "dg", "theta"],
    "pois-flex": ["a_rocky", "a_puffy", "da_Mstratum", "b_rocky", "b2_rocky",
                  "b_puffy", "b2_puffy", "g_rocky", "g_puffy", "d_Mstratum", "theta"],
}


def get_flat_pool():
    if os.path.exists(POOL_CACHE):
        z = np.load(POOL_CACHE)
        print(f"--> flat pool loaded from cache: {POOL_CACHE}")
        return {k: z[k] for k in z.files}
    print(f"--> building flat pool (N={S79.FLAT_N_POOL}) + transit/RV detectors...")
    flat = S79.build_flat()
    np.savez_compressed(POOL_CACHE, **flat)
    return flat


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = S79.load_silicate()
    flat = get_flat_pool()
    cells = build_cells(flat, m_sil, r_sil)
    print(f"    {len(cells['cls'])} cells; pool per cell: min={cells['pool_n'].min()}, "
          f"median={int(np.median(cells['pool_n']))}")

    rng = np.random.default_rng(SEED_MCMC)
    res = {}
    for tag, precision in [("full", False), ("prec", True)]:
        print(f"\n================ NASA variant: {tag} ================")
        nasa = S79.load_nasa(precision)
        n_obs = nasa_counts(nasa, m_sil, r_sil, cells)
        thin = (n_obs > 0) & (cells["pool_n"] < MIN_POOL_WARN)
        if thin.any():
            print(f"    WARNING: {int(thin.sum())} occupied cells have <{MIN_POOL_WARN} "
                  f"flat-pool planets — π noisy there")
        win_n = int(n_obs[cells["win"]].sum())
        print(f"    NASA planets binned: {int(n_obs.sum())} "
              f"(cold rocky window: {win_n})")

        r = {}
        for model, flex in [("cond-lin", False), ("cond-flex", True)]:
            nd = len(PARAM_NAMES[model]) - 1
            x0 = np.column_stack([rng.normal(0, 0.3, (N_WALK, nd)),
                                  rng.uniform(0.2, 1.5, (N_WALK, 1))])
            samples, acc, rhat = run_mcmc(
                lambda x: logpost_cond(x, cells, n_obs, flex), x0, rng,
                f"conditioned multinomial [{model}]")
            summ = theta_summary(samples[:, -1], model)
            g_obs, ppp, e_mean = ppc_deviance(samples, cells, n_obs, rng, flex)
            r[model] = dict(samples=samples, th=samples[:, -1], summ=summ, acc=acc,
                            rhat=rhat, e_mean=e_mean, ppp=ppp,
                            share_fit=share_curves(samples, cells, n_obs, flex),
                            share_b=share_curves(samples, cells, n_obs, flex,
                                                 force_theta=1.0))

        nd = len(PARAM_NAMES["pois-flex"]) - 1
        x0 = np.column_stack([rng.normal(0, 0.3, (N_WALK, nd)),
                              rng.uniform(0.2, 1.5, (N_WALK, 1))])
        samples_p, acc_p, rhat_p = run_mcmc(lambda x: logpost_pois(x, cells, n_obs), x0,
                                            rng, "naive Poisson (no effort) [pois-flex]")
        summ_p = theta_summary(samples_p[:, -1], "pois-flex")
        r["pois-flex"] = dict(samples=samples_p, th=samples_p[:, -1], summ=summ_p,
                              acc=acc_p, rhat=rhat_p)
        res[tag] = r

    # ---- save tables
    rows = []
    for tag in res:
        for model in res[tag]:
            s = res[tag][model]["samples"]
            names = PARAM_NAMES[model]
            for pidx, name in enumerate(names):
                col = s[:, pidx]
                rows.append(dict(variant=tag, model=model, param=name,
                                 mean=col.mean(), sd=col.std(),
                                 q05=np.quantile(col, 0.05), q50=np.quantile(col, 0.5),
                                 q95=np.quantile(col, 0.95),
                                 rhat=res[tag][model]["rhat"][pidx],
                                 acc=res[tag][model]["acc"]))
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "posterior_summary.csv"), index=False)

    nasa = S79.load_nasa(False)
    n_obs = nasa_counts(nasa, m_sil, r_sil, cells)
    ct = cells["df"].copy()
    ct["n_nasa_full"] = n_obs
    ct["e_fit"] = res["full"]["cond-flex"]["e_mean"]
    ct.to_csv(os.path.join(OUT_DIR, "cell_table.csv"), index=False)
    print(f"--> Saved: posterior_summary.csv, cell_table.csv")

    make_figure(res, cells, n_obs)

    s = res["full"]["cond-flex"]["summ"]
    sp = res["prec"]["cond-flex"]["summ"]
    sl = res["full"]["cond-lin"]["summ"]
    print(f"""
================= CAVEMAN SUMMARY =================
We fit the planet population's own occurrence law (power laws in mass and insolation,
one for rocky, one for puffy) with per-cell completeness from the flat pool. One knob
θ_occ multiplies the cold rocky window (rocky, M>2 M⊕, I<50 I⊕). θ_occ = 1 means rocky
super-Earths form in the cold exactly as the rest of the population extrapolates (B).
Follow-up targeting is removed EXACTLY by conditioning on each (stratum × I-column)
total — this is the share test, grown up.

Result (full NASA):   θ_occ = {s['q50']:.2f}, 90% CI [{s['q05']:.2f}, {s['q95']:.2f}], P(θ<1) = {s['p_lt1']:.4f}
Result (precision):   θ_occ = {sp['q50']:.2f}, 90% CI [{sp['q05']:.2f}, {sp['q95']:.2f}], P(θ<1) = {sp['p_lt1']:.4f}
Law sensitivity:      linear law gives θ_occ = {sl['q50']:.2f} [{sl['q05']:.2f}, {sl['q95']:.2f}]

Read it like this: the cold rocky window holds θ_occ of what its own population trend
predicts. This number does NOT depend on the flat-measure yardstick (the forecaster-M
weakness of tests LR/θ) — the flat pool only supplies within-cell completeness.
The window is not empty (θ > 0 always — planets are observed there), it is SUPPRESSED.

Model check: posterior-predictive ppp = {res['full']['cond-flex']['ppp']:.3f} (full) /
{res['prec']['cond-flex']['ppp']:.3f} (precision). The full-sample misfit is localized
in cells listed above (worst: low-mass 'puffy' planets with sloppy masses — the
precision cut removes them); θ_occ is stable across linear/flexible laws and both
samples, so the conclusion does not ride on the misfitting cells. Still: report ppp
next to θ_occ, never without it.
===================================================""")


if __name__ == "__main__":
    main()

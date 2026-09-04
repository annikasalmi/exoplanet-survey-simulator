"""
two_universe_puffy_overlap.py — can a PUFFY-ONLY universe be told apart
from a ROCKY+PUFFY one after the transit+RV pipelines?

Two universes (per the silicate boundary silicon_curve.ddat; above = puffy, on/below = rocky):
    A  "puffy only"     : planets only above the silicate M-R curve
    B  "rocky + puffy"   : the full physical population (rocky + puffy, natural mix)

For each Monte-Carlo realization we draw the SAME total number of planets, push them
through TRANSIT (Kepler) AND RV (best of HARPS+NIRPS), keep those detected by BOTH
(i.e. the planets that would appear on an M-R diagram), and record scalar
discriminators:
    * N_detected            (how many land on the M-R diagram)
    * mean detected radius   (where they sit in radius)

Repeating K times gives a sampling distribution of each discriminator for A and for
B. Their OVERLAP answers the question: how often would a puffy-only universe produce
a detected sample indistinguishable from a rocky+puffy one? Large overlap => the
detected sample cannot tell you whether the rocky planets exist.

Run under BOTH population priors (the user's choice):
    flat  : flat-parameter control universe (run/ppop/uniform_generator.py)
    ppop  : realistic P-Pop occurrence (cached Gaia-60pc Kepler catalog)

CAVEATS baked into the labels (see the chat critique):
  * the overlap depends on survey size (N drawn) — it shrinks as the sample grows;
  * requiring transit AND RV strongly selects big/massive planets, so this measures
    the PIPELINE's power to reveal the rocky population, not an intrinsic truth;
  * counts are Poisson, so we use the empirical MC distribution + a non-parametric
    overlap coefficient (OVL) and AUC, not a forced Gaussian.

Run:
    python scripts/two_universe_puffy_overlap.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from tools.paths import LIFESIM_OUTER_DIR, SILICON_CURVE
ROOT = Path(LIFESIM_OUTER_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run.flat_universe.uniform_generator import get_or_build_catalog, UNIFORM_OUT_DIR, generate_flat_catalog
from run.ppop.flat_detect import run_kepler, run_rv_best

# ---------------- settings ----------------
SILICATE_CURVE = Path(SILICON_CURVE)
PPOP_CATALOG = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"
OUT_DIR = os.path.join(ROOT, "my_outputs", "two_universe_puffy_overlap")

FLAT_N_POOL = 300000        # flat pool size (built once)
K_REALIZATIONS = 500        # Monte-Carlo universes per population
TARGET_DETECTED = 80        # tune N-drawn so each universe yields ~this many detected
RNG_SEED = 0

BOX = dict(r_lo=0.5, r_hi=2.2, m_lo=0.1, m_hi=12.0, f_lo=1e-2, f_hi=1e4)
RADIUS_HIST_BINS = np.linspace(0.5, 2.2, 18)


# ---------------- silicate boundary ----------------

def load_silicate():
    d = np.loadtxt(SILICATE_CURVE, comments="#")
    m, r = d[:, 0].astype(float), d[:, 1].astype(float)
    order = np.argsort(m)
    return m[order], r[order]


def is_puffy(mass, radius, m_sil, r_sil) -> np.ndarray:
    """Puffy = radius ABOVE the silicate curve at that mass (lower density)."""
    rc = np.interp(np.asarray(mass, float), m_sil, r_sil)
    return np.asarray(radius, float) > rc


# ---------------- pools (drawn + detected once, then bootstrapped) ----------------

def _box(df: pd.DataFrame) -> pd.DataFrame:
    r = pd.to_numeric(df["radius_p"], errors="coerce")
    m = pd.to_numeric(df["mass_p"], errors="coerce")
    f = pd.to_numeric(df.get("flux_p", np.nan), errors="coerce")
    keep = (r.between(BOX["r_lo"], BOX["r_hi"]) & m.between(BOX["m_lo"], BOX["m_hi"]))
    if f.notna().any():
        keep = keep & f.between(BOX["f_lo"], BOX["f_hi"])
    return df[keep].copy()


def build_pool(population: str, m_sil, r_sil) -> pd.DataFrame:
    """Return a pool with columns radius_p, mass_p, puffy(bool), det_both(bool)."""
    if population == "flat":
        pool = generate_flat_catalog(n_planets=FLAT_N_POOL, seed=RNG_SEED)
        pool = _box(pool)
        transit_det = run_kepler(pool)["detected"].to_numpy(bool)        # transit = Kepler
    elif population == "ppop":
        cols = ["radius_p", "mass_p", "p_orb", "inc_p", "ecc_p", "semimajor_p",
                "radius_s", "mass_s", "temp_s", "teff_s", "distance_s", "flux_p",
                "l_sun", "stype", "detected"]
        df = pd.read_csv(PPOP_CATALOG, usecols=lambda c: c in cols, low_memory=False)
        pool = _box(df)
        transit_det = pd.to_numeric(pool["detected"], errors="coerce").fillna(0).astype(bool).to_numpy()
    else:
        raise ValueError(population)

    rv = run_rv_best(pool, mag_target=12.0)
    rv_det = rv["detected"].to_numpy(bool)

    out = pd.DataFrame({
        "radius_p": pd.to_numeric(pool["radius_p"].to_numpy(), errors="coerce"),
        "mass_p": pd.to_numeric(pool["mass_p"].to_numpy(), errors="coerce"),
    })
    out["puffy"] = is_puffy(out["mass_p"], out["radius_p"], m_sil, r_sil)
    out["det_both"] = transit_det & rv_det
    out = out.dropna(subset=["radius_p", "mass_p"]).reset_index(drop=True)
    print(f"  [{population}] pool={len(out):,}  puffy={out['puffy'].mean():.1%}  "
          f"det_both(rate)={out['det_both'].mean():.3%}  "
          f"det_both rocky={int((out['det_both']&~out['puffy']).sum())}  "
          f"puffy={int((out['det_both']&out['puffy']).sum())}")
    return out


# ---------------- Monte-Carlo ----------------

def monte_carlo(pool: pd.DataFrame, rng):
    full_idx = np.arange(len(pool))
    puffy_idx = full_idx[pool["puffy"].to_numpy()]
    det = pool["det_both"].to_numpy(bool)
    rad = pool["radius_p"].to_numpy(float)

    rate = max(det.mean(), 1e-6)
    n_draw = int(TARGET_DETECTED / rate)   # same N for both universes

    rec = {"A_N": [], "B_N": [], "A_meanR": [], "B_meanR": []}
    histsA, histsB = [], []
    for _ in range(K_REALIZATIONS):
        b = rng.choice(full_idx, size=n_draw, replace=True)     # universe B: full mix
        a = rng.choice(puffy_idx, size=n_draw, replace=True)    # universe A: puffy only
        bd, ad = b[det[b]], a[det[a]]
        rec["B_N"].append(len(bd)); rec["A_N"].append(len(ad))
        rec["B_meanR"].append(rad[bd].mean() if len(bd) else np.nan)
        rec["A_meanR"].append(rad[ad].mean() if len(ad) else np.nan)
        histsB.append(np.histogram(rad[bd], bins=RADIUS_HIST_BINS)[0])
        histsA.append(np.histogram(rad[ad], bins=RADIUS_HIST_BINS)[0])
    out = {k: np.array(v, float) for k, v in rec.items()}
    out["n_draw"] = n_draw
    out["histA"] = np.array(histsA, float)
    out["histB"] = np.array(histsB, float)
    return out


# ---------------- overlap metrics ----------------

def ovl_coeff(a, b, bins=60):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    if hi <= lo:
        return 1.0
    edges = np.linspace(lo, hi, bins + 1)
    pa, _ = np.histogram(a, bins=edges, density=True)
    pb, _ = np.histogram(b, bins=edges, density=True)
    w = np.diff(edges)
    return float(np.sum(np.minimum(pa, pb) * w))


def auc_sep(a, b):
    """AUC = P(B > A); 0.5 = indistinguishable, 1 = perfectly separated."""
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    allv = np.concatenate([a, b])
    ranks = pd.Series(allv).rank().to_numpy()
    ra = ranks[:len(a)].sum()
    u = ra - len(a) * (len(a) + 1) / 2.0
    auc = 1.0 - u / (len(a) * len(b))     # P(B>A)
    return float(auc)


def cohen_d(a, b):
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    sp = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return float((b.mean() - a.mean()) / sp) if sp > 0 else np.nan


# ---------------- plotting ----------------

def panel_hist(ax, a, b, xlabel, title):
    lo, hi = np.nanmin([np.nanmin(a), np.nanmin(b)]), np.nanmax([np.nanmax(a), np.nanmax(b)])
    edges = np.linspace(lo, hi, 40)
    ax.hist(a, bins=edges, alpha=0.55, color="tab:orange", label="A: puffy only", density=True)
    ax.hist(b, bins=edges, alpha=0.55, color="tab:blue", label="B: rocky + puffy", density=True)
    ovl, auc, d = ovl_coeff(a, b), auc_sep(a, b), cohen_d(a, b)
    ax.set_title(f"{title}\nOVL={ovl:.2f}  AUC={auc:.2f}  d={d:+.2f}")
    ax.set_xlabel(xlabel); ax.set_ylabel("MC density")
    ax.legend(fontsize=8)
    return ovl, auc, d


def panel_radius_hist(ax, mc):
    centers = 0.5 * (RADIUS_HIST_BINS[:-1] + RADIUS_HIST_BINS[1:])
    for hists, color, lab in [(mc["histA"], "tab:orange", "A: puffy only"),
                              (mc["histB"], "tab:blue", "B: rocky + puffy")]:
        mean, sd = hists.mean(0), hists.std(0)
        ax.plot(centers, mean, color=color, lw=2, label=lab)
        ax.fill_between(centers, mean - sd, mean + sd, color=color, alpha=0.2)
    ax.set_xlabel(r"Detected planet radius [$R_\oplus$]")
    ax.set_ylabel("Mean # detected per universe")
    ax.set_title("Detected-radius distribution (mean ± 1σ over MC)")
    ax.legend(fontsize=8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = load_silicate()
    rng = np.random.default_rng(RNG_SEED)

    populations = ["flat", "ppop"]
    fig, axes = plt.subplots(len(populations), 3, figsize=(19, 5.4 * len(populations)))
    if len(populations) == 1:
        axes = axes[None, :]

    summary = []
    for i, pop in enumerate(populations):
        print(f"--> Building {pop} pool + detectors...")
        pool = build_pool(pop, m_sil, r_sil)
        mc = monte_carlo(pool, rng)
        print(f"  [{pop}] N drawn/universe = {mc['n_draw']:,}  "
              f"<N_det A>={mc['A_N'].mean():.1f}  <N_det B>={mc['B_N'].mean():.1f}")

        o1 = panel_hist(axes[i, 0], mc["A_N"], mc["B_N"],
                        "# planets detected (transit AND RV)", f"[{pop}] # detected")
        o2 = panel_hist(axes[i, 1], mc["A_meanR"], mc["B_meanR"],
                        r"mean detected radius [$R_\oplus$]", f"[{pop}] mean detected radius")
        panel_radius_hist(axes[i, 2], mc)
        axes[i, 2].set_title(f"[{pop}] " + axes[i, 2].get_title())

        summary.append((pop, mc["n_draw"], mc["A_N"].mean(), mc["B_N"].mean(),
                        o1[0], o1[1], o2[0], o2[1]))

    fig.suptitle(
        "Two-universe test: can a PUFFY-ONLY universe (A) be told apart from ROCKY+PUFFY (B)\n"
        "after Transit(Kepler) AND RV(best HARPS,NIRPS)?  OVL=overlap (1=identical), "
        "AUC=separability (0.5=indistinguishable)",
        fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_png = os.path.join(OUT_DIR, "two_universe_overlap.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved figure: {out_png}")

    lines = ["population  N_drawn  <N_det A>  <N_det B>  OVL(N)  AUC(N)  OVL(meanR)  AUC(meanR)"]
    for pop, nd, na, nb, ovlN, aucN, ovlR, aucR in summary:
        lines.append(f"{pop:<10} {nd:>7}  {na:>9.1f}  {nb:>9.1f}  {ovlN:>6.2f}  {aucN:>6.2f}"
                     f"  {ovlR:>9.2f}  {aucR:>9.2f}")
    txt = "\n".join(lines)
    (Path(OUT_DIR) / "summary.txt").write_text(txt, encoding="utf-8")
    print("\n" + txt)


if __name__ == "__main__":
    main()
DOWNLOAD_NASA_DATA = False  # Set to True to download fresh data, False to use local CSV

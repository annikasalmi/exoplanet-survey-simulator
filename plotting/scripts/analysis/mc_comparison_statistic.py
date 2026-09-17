"""Paper MC figure (mc_comparison_statistic_<N_DRAWS>.png): distribution of x_k = (f_k - f_obs)^2 /
sigma_obs^2 for I<10, I<50, I>50 (M>2), each draw a NASA-sized (7/27/75) mock survey with the
sample's 25%/8% errors, using bayesian_cold_rocky_desert.py's machinery.
Run: [N_DRAWS=500] python plotting/scripts/analysis/mc_comparison_statistic.py
"""

import os
import time

from plotting.scripts.analysis import bayesian_cold_rocky_desert as bayes

import numpy as np
import matplotlib.pyplot as plt

from tools.paths import ANALYSIS_DIR, PAPER_FIGURES_DIR

OUT_DIR = os.path.join(ANALYSIS_DIR, "mc_comparison_statistic")

MC_POOL_SIZE = 2_000_000
MC_CHUNK_SIZE = 2_000_000
N_DRAWS = int(os.environ.get("N_DRAWS", "5000"))

LABELS = {"rocky_formation": "Sub-Neptune + Super-Earth",
          "escape_only": "Sub-Neptune"}

# precision-cut ceilings AS the noise model: every planet in the sample passed
# M +-25% / R +-8%, so use those limits as a uniform fractional error — no
# per-planet published bars needed, only the N planets per insolation bin. The
# same errors are applied to the simulated planets, so both sides are measured alike.
MASS_CUT_ERR, RAD_CUT_ERR = 0.25, 0.08


def frac_draws(u, lo, hi, rng, m_sil, r_sil, n_obs, n_rep=N_DRAWS):
    """Per-draw volatile fractions f_k^(t) of a mock survey the size of NASA's: noise the bin's
    detected planets with the NASA sample's 25%/8% errors, apply M>2 to the noisy masses, then
    keep a random n_obs of the survivors, so the spread is what an n_obs-planet sample would see."""
    sel = u["det"] & (u["flux"] >= lo) & (u["flux"] < hi)
    m0, r0 = u["mass"][sel], u["radius"][sel]
    out = []
    for _ in range(n_rep):
        mo = m0 * np.exp(rng.normal(0.0, MASS_CUT_ERR, m0.size))
        ro = r0 * np.exp(rng.normal(0.0, RAD_CUT_ERR, r0.size))
        k = np.flatnonzero(mo > bayes.MASS_MIN)
        if k.size < n_obs:
            continue
        pick = rng.choice(k, n_obs, replace=False)
        out.append(float(bayes.is_volatile(mo[pick], ro[pick], m_sil, r_sil).mean()))
    return np.array(out)


def nasa_frac_draws(nasa, lo, hi, rng, m_sil, r_sil, n_rep=N_DRAWS):
    """Per-draw observed volatile fractions f_obs^(t) from flat 25%/8% (lognormal) noise on
    the bin's own N planets; M>2 cut reapplied per draw so boundary planets flip in/out."""
    sel = (nasa["ins"] >= lo) & (nasa["ins"] < hi)
    m, r = nasa["m"][sel], nasa["r"][sel]
    fr = np.empty(n_rep)
    for t in range(n_rep):
        mb = m * np.exp(rng.normal(0.0, MASS_CUT_ERR, m.size))
        rb = r * np.exp(rng.normal(0.0, RAD_CUT_ERR, r.size))
        k = mb > bayes.MASS_MIN
        fr[t] = bayes.is_volatile(mb[k], rb[k], m_sil, r_sil).mean() if k.any() else np.nan
    return fr[np.isfinite(fr)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    m_sil, r_sil = bayes.load_silicate()
    nasa = bayes.load_nasa(precision=True)

    print("--> building Otegi pool (rocky_formation; escape_only derived from it)")
    otegi = bayes.make_pool(
        "powerlaw", pool_size=MC_POOL_SIZE, chunk_size=MC_CHUNK_SIZE, cache_dir=OUT_DIR,
        mr_C=bayes.OTEGI_C, mr_beta=bayes.OTEGI_BETA,
        mass_scatter_dex=bayes.OTEGI_SCATTER,
    )
    rocky_true = ~bayes.is_volatile(otegi["mass"], otegi["radius"], m_sil, r_sil)
    keep = ~(rocky_true & (otegi["mass"] > bayes.MASS_MIN))
    univ = {"rocky_formation": otegi,
            "escape_only": {k: (v[keep] if isinstance(v, np.ndarray) else v)
                            for k, v in otegi.items()}}

    plt.rcParams.update({"font.size": 24, "axes.titlesize": 28, "axes.labelsize": 28,
                         "xtick.labelsize": 24, "ytick.labelsize": 24, "legend.fontsize": 21})
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5), layout="constrained")
    for ax, (blabel, lo, hi) in zip(axes, bayes.INSOL_BINS):
        k_obs, n_obs = bayes.nasa_bin(nasa, lo, hi, bayes.MASS_MIN, m_sil, r_sil)
        f_obs = k_obs / n_obs
        fobs_t = nasa_frac_draws(nasa, lo, hi, rng, m_sil, r_sil)  # sets sigma_obs only
        sig_obs = float(fobs_t.std())
        print(f"[{blabel}] NASA v/n = {k_obs}/{n_obs} = {f_obs:.3f} +- {sig_obs:.3f}")
        stats, points = {}, {}
        for key in ("rocky_formation", "escape_only"):
            fk = frac_draws(univ[key], lo, hi, rng, m_sil, r_sil, n_obs)
            stats[key] = ((fk - f_obs) / sig_obs) ** 2   # vs the raw observed number
            points[key] = ((fk.mean() - f_obs) / sig_obs) ** 2
        # an n_obs-planet survey can only give sqrt(x_k) = m * step for integer m. Put edges
        # halfway between allowed values (so no bin is empty), merging neighbours until each
        # bin is >= 1/20 of the range (so the narrow cells near 0 don't spike the density).
        step = 1.0 / (n_obs * sig_obs)
        m_max = int(np.ceil(np.sqrt(max(s.max() for s in stats.values())) / step)) + 1
        lattice = ((np.arange(m_max) + 0.5) * step) ** 2
        min_width = lattice[-1] / 20
        edges = [0.0]
        for e in lattice:
            if e - edges[-1] >= min_width:
                edges.append(e)
        if edges[-1] < lattice[-1]:
            edges[-1] = lattice[-1]
        for key in ("rocky_formation", "escape_only"):
            c = "C0" if key == "rocky_formation" else "C1"
            ax.hist(stats[key], bins=edges, color=c, alpha=0.45, density=True)
            ax.axvline(points[key], color=c, ls="--", lw=2, label=LABELS[key])
            print(f"    {key:<16} x_k point = {points[key]:.2f}  "
                  f"(draws: {stats[key].mean():.2f} +- {stats[key].std():.2f})")
        ax.set_xlim(left=-0.05)
        ax.set_title(f"{blabel} (M>2) $I_\\oplus$")
        ax.set_xlabel(r"$x_k = (f_k - f_{\rm obs})^2\,/\,\hat\sigma_{\rm obs}^2$")
        ax.set_ylabel("Probability density")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.15)

    fig.suptitle("MCMC runs of distribution compared to observed exoplanets", fontsize=30)
    out = os.path.join(OUT_DIR, f"mc_comparison_statistic_{N_DRAWS}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    os.makedirs(PAPER_FIGURES_DIR, exist_ok=True)
    fig.savefig(os.path.join(PAPER_FIGURES_DIR, os.path.basename(out)), dpi=150, bbox_inches="tight")
    print(f"--> Saved: {out}  ({time.time()-t0:.0f}s), plus a copy in {PAPER_FIGURES_DIR}")


if __name__ == "__main__":
    main()

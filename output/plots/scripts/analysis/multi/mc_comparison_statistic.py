"""Paper MC figure (mc_comparison_statistic_<N_DRAWS>.png): distribution of sqrt(x_k) = |f_k - f_obs| /
sigma_obs for I<10, I<50, I>50 (M>2), using bayesian_cold_rocky_desert.py's machinery.
Run: [N_DRAWS=500] python output/plots/scripts/analysis/multi/mc_comparison_statistic.py
"""

import importlib.util
import os
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "bayes_cold_rocky_desert", _HERE / "bayesian_cold_rocky_desert.py")
S41 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S41)

import numpy as np
import matplotlib.pyplot as plt

from tools.paths import ANALYSIS_DIR

OUT_DIR = os.path.join(ANALYSIS_DIR, "mc_comparison_statistic")

# ---- overrides: small pool, own cache dir ----
S41.FLAT_N_POOL = 2_000_000
S41.CHUNK = 2_000_000
S41._out_dir = lambda: OUT_DIR
N_DRAWS = int(os.environ.get("N_DRAWS", "5000"))

LABELS = {"rocky_formation": "Sub-Neptune + Super-Earth",
          "escape_only": "Sub-Neptune"}

# precision-cut ceilings AS the noise model: every planet in the sample passed
# M +-25% / R +-8%, so use those limits as a uniform fractional error — no
# per-planet published bars needed, only the N planets per insolation bin.
MASS_CUT_ERR, RAD_CUT_ERR = 0.25, 0.08


def frac_draws(u, lo, hi, rng, m_sil, r_sil, n_rep=N_DRAWS):
    """Per-draw detected volatile fractions f_k^(t) (script 41 predicted_frac, array form)."""
    sel = u["det"] & (u["flux"] >= lo) & (u["flux"] < hi)
    m0, r0 = u["mass"][sel], u["radius"][sel]
    out = []
    for _ in range(n_rep):
        mo = m0 * np.exp(rng.normal(0.0, S41.MASS_FRAC_ERR, m0.size))
        ro = r0 * np.exp(rng.normal(0.0, S41.RAD_FRAC_ERR, r0.size))
        k = mo > S41.MASS_MIN
        if k.sum() < 5:
            continue
        out.append(float(S41.is_volatile(mo[k], ro[k], m_sil, r_sil).mean()))
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
        k = mb > S41.MASS_MIN
        fr[t] = S41.is_volatile(mb[k], rb[k], m_sil, r_sil).mean() if k.any() else np.nan
    return fr[np.isfinite(fr)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    m_sil, r_sil = S41.load_silicate()
    nasa = S41.load_nasa(precision=True)

    print("--> building Otegi pool (rocky_formation; escape_only derived from it)")
    otegi = S41.make_pool("powerlaw", mr_C=S41.OTEGI_C, mr_beta=S41.OTEGI_BETA,
                          mass_scatter_dex=S41.OTEGI_SCATTER)
    rocky_true = ~S41.is_volatile(otegi["mass"], otegi["radius"], m_sil, r_sil)
    keep = ~(rocky_true & (otegi["mass"] > S41.MASS_MIN))
    univ = {"rocky_formation": otegi,
            "escape_only": {k: (v[keep] if isinstance(v, np.ndarray) else v)
                            for k, v in otegi.items()}}

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    for ax, (blabel, lo, hi) in zip(axes, S41.INSOL_BINS):
        k_obs, n_obs = S41.nasa_bin(nasa, lo, hi, S41.MASS_MIN, m_sil, r_sil)
        f_obs = k_obs / n_obs
        fobs_t = nasa_frac_draws(nasa, lo, hi, rng, m_sil, r_sil)  # sets sigma_obs only
        sig_obs = float(fobs_t.std())
        print(f"[{blabel}] NASA v/n = {k_obs}/{n_obs} = {f_obs:.3f} +- {sig_obs:.3f}")
        stats, points = {}, {}
        for key in ("rocky_formation", "escape_only"):
            fk = frac_draws(univ[key], lo, hi, rng, m_sil, r_sil)
            stats[key] = np.abs(fk - f_obs) / sig_obs   # vs the raw observed number
            points[key] = abs(fk.mean() - f_obs) / sig_obs
        xmax = max(s.max() for s in stats.values()) * 1.06
        edges = np.linspace(0.0, xmax, 60)
        bw = edges[1] - edges[0]
        for key in ("rocky_formation", "escape_only"):
            c = "C0" if key == "rocky_formation" else "C1"
            ax.hist(stats[key], bins=edges, color=c, alpha=0.45)
            ax.axvline(points[key], color=c, ls="--", lw=2,
                       label=f"{LABELS[key]}: $\\sqrt{{x_k}}$={points[key]:.2f}")
            print(f"    {key:<16} sqrt(x_k) point = {points[key]:.2f}  "
                  f"(draws: {stats[key].mean():.2f} +- {stats[key].std():.2f})")
        # NASA vs itself: redraw NASA within its own error bars -> |f^(t)-f_obs|/sigma_obs is
        # half-normal with unit sigma ON THIS AXIS (the axis is in units of sigma_obs); the
        # insolation dependence is sigma_obs itself, quoted in the legend per panel.
        xs = np.linspace(0.0, xmax, 400)
        ax.plot(xs, 2.0 * N_DRAWS * bw * np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi),
                color="C2", lw=2.5,
                label=f"NASA vs itself: half-normal, $\\hat\\sigma_{{\\rm obs}}$={sig_obs:.3f}")
        ax.set_xlim(left=-0.05)
        ax.set_title(f"{blabel} (M>2) $I_\\oplus$", fontsize=13)
        ax.set_xlabel(r"$\sqrt{x_k} = |f_k - f_{\rm obs}|\,/\,\hat\sigma_{\rm obs}$")
        ax.set_ylabel(f"number of MC draws (of {N_DRAWS:,})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.15)

    fig.suptitle(
        "Monte Carlo distribution of the comparison statistic on the ROOT scale, "
        r"$\sqrt{x_k}$ — Kepler transit + RV, precision-cut sample, $M>2\,M_\oplus$"
        "\nhistograms: per-draw $|f_k^{(t)} - f_{\\rm obs}|/\\hat\\sigma_{\\rm obs}$ against the "
        "raw observed fraction;  "
        "dashed: the point value;  green: NASA against itself — half-normal, "
        "$\\hat\\sigma_{\\rm obs}$ from the precision-cut limits ($M\\pm25\\%$, $R\\pm8\\%$) "
        "on each bin's own N planets",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(OUT_DIR, f"mc_comparison_statistic_{N_DRAWS}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"--> Saved: {out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

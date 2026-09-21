"""Paper MC figure (mc_comparison_statistic_<N_DRAWS>.png): distribution of x_k = (f_k - f_obs)^2 /
sigma_obs^2 for I<10, I<50, I>50 (M>2), each draw a NASA-sized (7/27/75) mock survey with the
sample's 25%/8% errors, using the shared science.comparison machinery.
Run: [N_DRAWS=500] python plotting/scripts/analysis/mc_comparison_statistic.py
"""

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from science.physics import (
    SUPER_EARTH_MIN_MASS,
    load_mass_radius_curve,
)
from science.catalogs import COMPARISON_PARAMETER_BOX, load_measured_planets
from science.comparison import (
    INSOLATION_BINS,
    mock_survey_volatile_fractions,
    observed_volatile_count,
    observed_volatile_fraction_draws,
)
from science.telescopes.detection import make_detected_pool, split_universes
from science.statistics import NASA_MEASUREMENT_ERROR
from tools.paths import ANALYSIS_DIR, PAPER_FIGURES_DIR, PSCOMPPARS_CSV
from tools.plotting_constants import PAPER_STYLE

OUT_DIR = os.path.join(ANALYSIS_DIR, "mc_comparison_statistic")

MC_POOL_SIZE = 2_000_000
MC_CHUNK_SIZE = 2_000_000
N_DRAWS = int(os.environ.get("N_DRAWS", "5000"))

LABELS = {"rocky_formation": "Sub-Neptune + Super-Earth",
          "escape_only": "Sub-Neptune"}

# The sample's precision cuts (M +-25%, R +-8%) as the noise model, on NASA and simulated planets alike.
COMPARISON_ERROR = NASA_MEASUREMENT_ERROR


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    m_sil, r_sil = load_mass_radius_curve()
    nasa = load_measured_planets(
        PSCOMPPARS_CSV,
        mass_bounds=(COMPARISON_PARAMETER_BOX["m_lo"], COMPARISON_PARAMETER_BOX["m_hi"]),
        radius_bounds=(COMPARISON_PARAMETER_BOX["r_lo"], COMPARISON_PARAMETER_BOX["r_hi"]),
        insolation_bounds=(COMPARISON_PARAMETER_BOX["f_lo"], COMPARISON_PARAMETER_BOX["f_hi"]),
        max_relative_error=NASA_MEASUREMENT_ERROR,
        missing_relative_error=NASA_MEASUREMENT_ERROR,
    )

    print("--> building superearths_supneptunes pool (rocky_formation; escape_only is only_subneptunes)")
    univ = split_universes(make_detected_pool(
        "superearths_supneptunes", pool_size=MC_POOL_SIZE, chunk_size=MC_CHUNK_SIZE,
        cache_dir=OUT_DIR, box=COMPARISON_PARAMETER_BOX))

    plt.rcParams.update(PAPER_STYLE)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5), layout="constrained")
    for ax, (blabel, lo, hi) in zip(axes, INSOLATION_BINS):
        k_obs, n_obs = observed_volatile_count(
            nasa, lo, hi, m_sil, r_sil, mass_min=SUPER_EARTH_MIN_MASS)
        f_obs = k_obs / n_obs
        fobs_t = observed_volatile_fraction_draws(
            nasa, lo, hi, m_sil, r_sil, rng, repeats=N_DRAWS,
            mass_min=SUPER_EARTH_MIN_MASS, error=COMPARISON_ERROR,
        )
        sig_obs = float(fobs_t.std())
        print(f"[{blabel}] NASA v/n = {k_obs}/{n_obs} = {f_obs:.3f} +- {sig_obs:.3f}")
        stats, points = {}, {}
        for key in ("rocky_formation", "escape_only"):
            fk = mock_survey_volatile_fractions(
                univ[key], lo, hi, m_sil, r_sil, rng, n_obs,
                repeats=N_DRAWS, mass_min=SUPER_EARTH_MIN_MASS,
                error=COMPARISON_ERROR,
            )
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
